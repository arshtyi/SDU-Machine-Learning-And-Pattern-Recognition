from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.datasets import make_blobs
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RANDOM_STATE = 0
CMAP = "rainbow"


@dataclass
class ModelResult:
    name: str
    train_accuracy: float
    test_accuracy: float
    depth_summary: str
    leaf_summary: str


def make_dataset() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs(
        n_samples=300,
        centers=4,
        random_state=RANDOM_STATE,
        cluster_std=1.0,
    )


def data_limits(
    x: np.ndarray, pad: float = 0.8
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = float(x[:, 0].min() - pad), float(x[:, 0].max() + pad)
    y_min, y_max = float(x[:, 1].min() - pad), float(x[:, 1].max() + pad)
    return (x_min, x_max), (y_min, y_max)


def make_bagging_classifier(
    *,
    n_estimators: int,
    max_samples: float,
    random_state: int,
    bootstrap: bool = True,
) -> BaggingClassifier:
    tree = DecisionTreeClassifier(random_state=random_state)
    kwargs = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "random_state": random_state,
        "bootstrap": bootstrap,
    }
    try:
        return BaggingClassifier(estimator=tree, **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=tree, **kwargs)


def draw_animal_decision_tree(output_path: Path) -> None:
    nodes = {
        "root": ("Is it bigger\nthan a breadbox?", (0.50, 0.90)),
        "yes": ("Does it have\nhorns?", (0.28, 0.66)),
        "no": ("Does it have\nfeathers?", (0.72, 0.66)),
        "bear": ("Bear", (0.14, 0.42)),
        "deer": ("Deer", (0.42, 0.42)),
        "hawk": ("Hawk", (0.60, 0.42)),
        "mouse": ("Mouse", (0.84, 0.42)),
    }
    edges = [
        ("root", "yes", "yes"),
        ("root", "no", "no"),
        ("yes", "bear", "no"),
        ("yes", "deer", "yes"),
        ("no", "hawk", "yes"),
        ("no", "mouse", "no"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_axis_off()
    for parent, child, label in edges:
        parent_xy = nodes[parent][1]
        child_xy = nodes[child][1]
        ax.annotate(
            "",
            xy=(child_xy[0], child_xy[1] + 0.065),
            xytext=(parent_xy[0], parent_xy[1] - 0.065),
            arrowprops={"arrowstyle": "->", "color": "#4B5563", "linewidth": 1.4},
        )
        ax.text(
            (parent_xy[0] + child_xy[0]) / 2,
            (parent_xy[1] + child_xy[1]) / 2 + 0.02,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="#374151",
        )
    for key, (text, xy) in nodes.items():
        is_leaf = key in {"bear", "deer", "hawk", "mouse"}
        ax.text(
            xy[0],
            xy[1],
            text,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold" if is_leaf else "normal",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#E8F5E9" if is_leaf else "#E3F2FD",
                "edgecolor": "#374151",
                "linewidth": 1.2,
            },
        )
    ax.set_title("A Simple Hand-Written Decision Tree", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dataset(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=y,
        s=50,
        cmap=CMAP,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("Two-Dimensional Four-Class Data")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize_classifier(
    model,
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes,
    *,
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
    mesh_steps: int = 300,
    point_size: int = 30,
) -> None:
    xlim, ylim = limits
    model.fit(x, y)

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], num=mesh_steps),
        np.linspace(ylim[0], ylim[1], num=mesh_steps),
    )
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    n_classes = len(np.unique(y))

    ax.contourf(
        xx,
        yy,
        z,
        alpha=0.30,
        levels=np.arange(n_classes + 1) - 0.5,
        cmap=CMAP,
        zorder=1,
    )
    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=y,
        s=point_size,
        cmap=CMAP,
        edgecolor="black",
        linewidth=0.25,
        zorder=3,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_depth_sequence(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    limits = data_limits(x)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for depth, ax in zip([1, 2, 3, 4], axes.ravel(), strict=True):
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        visualize_classifier(
            model,
            x,
            y,
            ax,
            title=f"Decision tree depth = {depth}",
            limits=limits,
        )
    fig.suptitle("First Four Levels of a Decision Tree", y=0.98, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_single_tree(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    visualize_classifier(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        x,
        y,
        ax,
        title="Unrestricted Decision Tree",
        limits=data_limits(x),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_random_subset_trees(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    rng = np.random.default_rng(RANDOM_STATE + 1)
    halves = np.array_split(rng.permutation(len(y)), 2)
    limits = data_limits(x)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for part_number, (indices, ax) in enumerate(
        zip(halves, axes, strict=True), start=1
    ):
        visualize_classifier(
            DecisionTreeClassifier(random_state=RANDOM_STATE + part_number),
            x[indices],
            y[indices],
            ax,
            title=f"Tree trained on random half {part_number}",
            limits=limits,
        )
    fig.suptitle("Two Trees Trained on Different Random Subsets", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_two_tree_ensemble(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    model = make_bagging_classifier(
        n_estimators=2,
        max_samples=0.5,
        random_state=RANDOM_STATE + 2,
        bootstrap=False,
    )
    visualize_classifier(
        model,
        x,
        y,
        ax,
        title="Two-Tree Subsample Ensemble",
        limits=data_limits(x),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bagging_classifier(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    model = make_bagging_classifier(
        n_estimators=100,
        max_samples=0.8,
        random_state=1,
    )
    visualize_classifier(
        model,
        x,
        y,
        ax,
        title="Bagging of 100 Decision Trees",
        limits=data_limits(x),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_random_forest_classifier(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    visualize_classifier(
        model,
        x,
        y,
        ax,
        title="Random Forest of 100 Trees",
        limits=data_limits(x),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_tree_collection(model) -> tuple[str, str]:
    if isinstance(model, DecisionTreeClassifier):
        return str(model.get_depth()), str(model.get_n_leaves())

    estimators = getattr(model, "estimators_", [])
    depths = [
        estimator.get_depth()
        for estimator in estimators
        if hasattr(estimator, "get_depth")
    ]
    leaves = [
        estimator.get_n_leaves()
        for estimator in estimators
        if hasattr(estimator, "get_n_leaves")
    ]
    if not depths or not leaves:
        return "-", "-"
    return (
        f"mean={np.mean(depths):.2f}, max={np.max(depths):.0f}",
        f"mean={np.mean(leaves):.2f}, max={np.max(leaves):.0f}",
    )


def evaluate_models(x: np.ndarray, y: np.ndarray, output_path: Path) -> list[ModelResult]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.35,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    models = [
        ("single decision tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        (
            "bagging classifier",
            make_bagging_classifier(
                n_estimators=100,
                max_samples=0.8,
                random_state=1,
            ),
        ),
        (
            "random forest",
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        ),
    ]

    results: list[ModelResult] = []
    for name, model in models:
        model.fit(x_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(x_train))
        test_accuracy = accuracy_score(y_test, model.predict(x_test))
        depth_summary, leaf_summary = summarize_tree_collection(model)
        results.append(
            ModelResult(
                name=name,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                depth_summary=depth_summary,
                leaf_summary=leaf_summary,
            )
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "train_accuracy",
                "test_accuracy",
                "depth_summary",
                "leaf_summary",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.name,
                    "train_accuracy": f"{result.train_accuracy:.6f}",
                    "test_accuracy": f"{result.test_accuracy:.6f}",
                    "depth_summary": result.depth_summary,
                    "leaf_summary": result.leaf_summary,
                }
            )
    return results


def write_experiment_summary(results: list[ModelResult], output_path: Path) -> None:
    lines = [
        "# Experiment 10 Summary",
        "",
        "| Model | Train accuracy | Test accuracy | Depth | Leaves |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.train_accuracy:.4f} | "
            f"{result.test_accuracy:.4f} | {result.depth_summary} | {result.leaf_summary} |"
        )
    lines.extend(
        [
            "",
            "The unrestricted tree tends to form jagged regions that follow sample noise.",
            "Bagging reduces the dependence on one particular sample draw.",
            "Random forests add feature-level randomness, producing a smoother and more stable boundary.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    x, y = make_dataset()
    draw_animal_decision_tree(OUTPUT_DIR / "animal_decision_tree.png")
    plot_dataset(x, y, OUTPUT_DIR / "blob_dataset.png")
    plot_depth_sequence(x, y, OUTPUT_DIR / "decision_tree_depths.png")
    plot_single_tree(x, y, OUTPUT_DIR / "decision_tree_full.png")
    plot_random_subset_trees(x, y, OUTPUT_DIR / "random_subset_trees.png")
    plot_two_tree_ensemble(x, y, OUTPUT_DIR / "two_tree_ensemble.png")
    plot_bagging_classifier(x, y, OUTPUT_DIR / "bagging_classifier.png")
    plot_random_forest_classifier(x, y, OUTPUT_DIR / "random_forest_classifier.png")

    results = evaluate_models(x, y, OUTPUT_DIR / "model_accuracy.csv")
    write_experiment_summary(results, OUTPUT_DIR / "summary.md")

    print("=== Experiment 10: Random Forest ===")
    print(f"samples={x.shape[0]}, features={x.shape[1]}, classes={len(np.unique(y))}")
    for result in results:
        print(
            f"{result.name:22s} train={result.train_accuracy:.4f} "
            f"test={result.test_accuracy:.4f} depth={result.depth_summary}"
        )
    print(f"outputs saved to: {OUTPUT_DIR}")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
