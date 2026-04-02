import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pypdf.generic import IndirectObject


@dataclass(frozen=True)
class PdfTask:
    pdf_path: Path
    source_group: str


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def collect_pdf_tasks(instruction_dir: Path) -> list[PdfTask]:
    experiment_dir = instruction_dir.parent
    project_root = experiment_dir.parent

    tasks: list[PdfTask] = []
    seen: set[Path] = set()

    # ../ 下除 instruction 外的每个目录中的 text/*.pdf
    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        if child.resolve() == instruction_dir.resolve():
            continue
        if not is_within(child, experiment_dir):
            continue

        text_dir = child / "text"
        if not text_dir.is_dir():
            continue
        if not is_within(text_dir, child):
            continue

        for pdf in sorted(text_dir.glob("*.pdf")):
            resolved = pdf.resolve()
            if resolved in seen:
                continue
            if not is_within(resolved, text_dir):
                continue
            seen.add(resolved)
            tasks.append(
                PdfTask(pdf_path=resolved, source_group=f"experiment::{child.name}")
            )

    # ../../text/*.pdf
    upper_text_dir = project_root / "text"
    if upper_text_dir.is_dir() and is_within(upper_text_dir, project_root):
        for pdf in sorted(upper_text_dir.glob("*.pdf")):
            resolved = pdf.resolve()
            if resolved in seen:
                continue
            if not is_within(resolved, upper_text_dir):
                continue
            seen.add(resolved)
            tasks.append(PdfTask(pdf_path=resolved, source_group="project_root::text"))

    tasks.sort(key=lambda x: str(x.pdf_path))
    return tasks


def _safe_get_font_name(font_obj) -> str | None:
    try:
        if isinstance(font_obj, IndirectObject):
            font_obj = font_obj.get_object()
        base_font = font_obj.get("/BaseFont")
        if base_font is None:
            return None
        return str(base_font)
    except Exception:
        return None


def extract_font_names(reader: PdfReader) -> list[str]:
    fonts: set[str] = set()
    for page in reader.pages:
        try:
            resources = page.get("/Resources")
            if isinstance(resources, IndirectObject):
                resources = resources.get_object()
            if not resources:
                continue

            font_dict = resources.get("/Font")
            if isinstance(font_dict, IndirectObject):
                font_dict = font_dict.get_object()
            if not font_dict:
                continue

            for _, font_obj in font_dict.items():
                name = _safe_get_font_name(font_obj)
                if name:
                    fonts.add(name)
        except Exception:
            continue
    return sorted(fonts)


def extract_pdf_metadata(pdf_path: Path) -> dict:
    result = {
        "file": str(pdf_path),
        "size_bytes": pdf_path.stat().st_size,
        "pages": None,
        "encrypted": None,
        "title": None,
        "author": None,
        "subject": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "modification_date": None,
        "copyright": None,
        "fonts": [],
        "extra_info": {},
        "error": None,
    }
    try:
        reader = PdfReader(str(pdf_path), strict=False)
        meta = reader.metadata

        result["pages"] = len(reader.pages)
        result["encrypted"] = bool(reader.is_encrypted)

        if meta is not None:
            result["title"] = getattr(meta, "title", None)
            result["author"] = getattr(meta, "author", None)
            result["subject"] = getattr(meta, "subject", None)
            result["creator"] = getattr(meta, "creator", None)
            result["producer"] = getattr(meta, "producer", None)
            result["creation_date"] = getattr(meta, "creation_date", None)
            result["modification_date"] = getattr(meta, "modification_date", None)

            raw_meta = {}
            for key, value in meta.items():
                raw_meta[str(key)] = str(value)
            result["extra_info"] = raw_meta
            result["copyright"] = (
                raw_meta.get("/Copyright")
                or raw_meta.get("Copyright")
                or raw_meta.get("/Rights")
                or raw_meta.get("Rights")
            )

        result["fonts"] = extract_font_names(reader)
    except (PdfReadError, OSError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"UnexpectedError: {type(exc).__name__}: {exc}"
    return result


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if value == "":
        return "N/A"
    return str(value)


def write_metadata_summary(summary_path: Path, entries: Iterable[dict]) -> None:
    lines: list[str] = []
    lines.append("# PDF Metadata Summary")
    lines.append("")
    lines.append("> Generated by `instruction/main.py`.")
    lines.append("")

    for idx, entry in enumerate(entries, start=1):
        lines.append(f"## {idx}. {Path(entry['file']).name}")
        lines.append("")
        lines.append(f"- File: `{entry['file']}`")
        lines.append(f"- Size (bytes): `{entry['size_bytes']}`")
        lines.append(f"- Pages: `{_fmt(entry['pages'])}`")
        lines.append(f"- Encrypted: `{_fmt(entry['encrypted'])}`")
        lines.append(f"- Title: `{_fmt(entry['title'])}`")
        lines.append(f"- Author: `{_fmt(entry['author'])}`")
        lines.append(f"- Subject: `{_fmt(entry['subject'])}`")
        lines.append(f"- Creator: `{_fmt(entry['creator'])}`")
        lines.append(f"- Producer: `{_fmt(entry['producer'])}`")
        lines.append(f"- CreationDate: `{_fmt(entry['creation_date'])}`")
        lines.append(f"- ModDate: `{_fmt(entry['modification_date'])}`")
        lines.append(f"- Copyright/Rights: `{_fmt(entry['copyright'])}`")

        fonts = entry.get("fonts", [])
        if fonts:
            lines.append("- Fonts:")
            for font_name in fonts:
                lines.append(f"  - `{font_name}`")
        else:
            lines.append("- Fonts: `N/A`")

        err = entry.get("error")
        if err:
            lines.append(f"- Metadata Read Error: `{err}`")

        extra = entry.get("extra_info", {})
        if extra:
            lines.append("- Extra Metadata Keys:")
            for key in sorted(extra.keys()):
                lines.append(f"  - `{key}`: `{extra[key]}`")

        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def build_marker_command(
    pdf_path: Path, output_dir: Path, prefer_uv: bool = True
) -> list[str]:
    cmd = [
        "marker_single",
        str(pdf_path),
        "--output_format",
        "markdown",
        "--output_dir",
        str(output_dir),
    ]
    if prefer_uv and shutil.which("uv"):
        return ["uv", "run", *cmd]
    return cmd


def convert_with_marker(
    task: PdfTask, output_root: Path, env: dict, prefer_uv: bool
) -> tuple[bool, str]:
    safe_group = (
        task.source_group.replace("::", "_").replace("/", "_").replace("\\", "_")
    )
    task_output_dir = output_root / safe_group
    task_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_marker_command(task.pdf_path, task_output_dir, prefer_uv=prefer_uv)
    try:
        subprocess.run(cmd, env=env, check=True)
        return True, ""
    except subprocess.CalledProcessError as exc:
        return False, f"CalledProcessError(exit_code={exc.returncode})"
    except FileNotFoundError as exc:
        return False, f"FileNotFoundError: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"UnexpectedError: {type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect PDF metadata and convert target PDFs to markdown with marker-pdf."
    )
    parser.add_argument(
        "--summary-file",
        default="pdf_metadata_summary.md",
        help="Metadata summary markdown filename (written under current script directory).",
    )
    parser.add_argument(
        "--output-root",
        default="marker_output",
        help="Output root directory for marker conversion (under current script directory).",
    )
    parser.add_argument(
        "--direct-marker",
        action="store_true",
        help="Use `marker_single` directly instead of `uv run marker_single`.",
    )
    args = parser.parse_args()

    instruction_dir = Path(__file__).resolve().parent
    summary_path = (instruction_dir / args.summary_file).resolve()
    output_root = (instruction_dir / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = collect_pdf_tasks(instruction_dir)
    if not tasks:
        print("未找到符合规则的 PDF 文件。")
        return 0

    use_cuda = torch.cuda.is_available()
    env = os.environ.copy()
    env["TORCH_DEVICE"] = "cuda" if use_cuda else "cpu"
    if use_cuda:
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}，将使用 TORCH_DEVICE=cuda")
    else:
        print("未检测到可用 CUDA，将回退为 TORCH_DEVICE=cpu")

    metadata_entries: list[dict] = []
    conversion_results: list[tuple[PdfTask, bool, str]] = []

    for task in tasks:
        print(f"[metadata] {task.pdf_path}")
        metadata_entries.append(extract_pdf_metadata(task.pdf_path))

    write_metadata_summary(summary_path, metadata_entries)
    print(f"metadata 汇总已写入: {summary_path}")

    prefer_uv = not args.direct_marker
    for task in tasks:
        print(f"[convert] {task.pdf_path}")
        ok, err = convert_with_marker(
            task, output_root=output_root, env=env, prefer_uv=prefer_uv
        )
        conversion_results.append((task, ok, err))
        if ok:
            print("  -> success")
        else:
            print(f"  -> failed: {err}")

    success_count = sum(1 for _, ok, _ in conversion_results if ok)
    fail_count = len(conversion_results) - success_count
    print("")
    print(
        f"总计: {len(conversion_results)} 个 PDF, 成功 {success_count}, 失败 {fail_count}"
    )
    if fail_count:
        print("失败列表:")
        for task, ok, err in conversion_results:
            if not ok:
                print(f"- {task.pdf_path}: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
