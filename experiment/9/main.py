from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RES_DIR = BASE_DIR / "res"
OUTPUT_DIR = BASE_DIR / "output"
RANDOM_STATE = 42

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


@dataclass
class TreeNode:
    prediction: int
    probability: float
    impurity: float
    samples: int
    positives: int
    feature_index: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeClassifier:
    """A small CART-style binary decision tree implemented from scratch."""

    def __init__(
        self,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int | None = 6,
        min_samples_split: int = 12,
        min_samples_leaf: int = 6,
        min_impurity_decrease: float = 1e-7,
    ) -> None:
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root: TreeNode | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.root = self._grow_tree(x, y.astype(int), depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("fit must be called before predict")
        return np.array([self._predict_one(row, self.root) for row in x], dtype=int)

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> int:
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            node = (
                node.left if row[node.feature_index] <= node.threshold else node.right
            )
            assert node is not None
        return node.prediction

    def _grow_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = int(y.size)
        positives = int(y.sum())
        probability = positives / n_samples
        impurity = float(self._impurity_scalar(positives, n_samples))
        prediction = int(probability >= 0.5)
        node = TreeNode(
            prediction=prediction,
            probability=probability,
            impurity=impurity,
            samples=n_samples,
            positives=positives,
        )

        reached_max_depth = self.max_depth is not None and depth >= self.max_depth
        pure = positives == 0 or positives == n_samples
        too_small = n_samples < self.min_samples_split
        if reached_max_depth or pure or too_small:
            return node

        split = self._best_split(x, y, parent_impurity=impurity)
        if split is None:
            return node
        feature_index, threshold, gain = split
        if gain <= self.min_impurity_decrease:
            return node

        left_mask = x[:, feature_index] <= threshold
        node.feature_index = feature_index
        node.threshold = threshold
        node.gain = gain
        node.left = self._grow_tree(x[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(x[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(
        self, x: np.ndarray, y: np.ndarray, parent_impurity: float
    ) -> tuple[int, float, float] | None:
        best_feature = -1
        best_threshold = 0.0
        best_gain = 0.0
        n_samples, n_features = x.shape
        total_pos = y.sum()

        for feature_index in range(n_features):
            order = np.argsort(x[:, feature_index], kind="mergesort")
            values = x[order, feature_index]
            labels = y[order]
            distinct_positions = np.nonzero(values[:-1] != values[1:])[0] + 1
            if distinct_positions.size == 0:
                continue

            left_n = distinct_positions
            right_n = n_samples - left_n
            valid = (left_n >= self.min_samples_leaf) & (
                right_n >= self.min_samples_leaf
            )
            if not np.any(valid):
                continue

            positions = distinct_positions[valid]
            left_n = left_n[valid]
            right_n = right_n[valid]
            cumsum_pos = np.cumsum(labels)
            left_pos = cumsum_pos[positions - 1]
            right_pos = total_pos - left_pos

            left_impurity = self._impurity_vector(left_pos, left_n)
            right_impurity = self._impurity_vector(right_pos, right_n)
            weighted_impurity = (
                left_n * left_impurity + right_n * right_impurity
            ) / n_samples
            gains = parent_impurity - weighted_impurity
            current_best = int(np.argmax(gains))
            current_gain = float(gains[current_best])

            if current_gain > best_gain:
                split_pos = int(positions[current_best])
                best_feature = feature_index
                best_threshold = float((values[split_pos - 1] + values[split_pos]) / 2)
                best_gain = current_gain

        if best_feature == -1:
            return None
        return best_feature, best_threshold, best_gain

    def _impurity_scalar(self, positives: int, n_samples: int) -> float:
        if n_samples == 0:
            return 0.0
        p = positives / n_samples
        if self.criterion == "gini":
            return 1.0 - p**2 - (1.0 - p) ** 2
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    def _impurity_vector(
        self, positives: np.ndarray, n_samples: np.ndarray
    ) -> np.ndarray:
        p = positives / n_samples
        if self.criterion == "gini":
            return 1.0 - p**2 - (1.0 - p) ** 2
        out = np.zeros_like(p, dtype=float)
        mask = (p > 0.0) & (p < 1.0)
        out[mask] = -(
            p[mask] * np.log2(p[mask]) + (1.0 - p[mask]) * np.log2(1.0 - p[mask])
        )
        return out


def load_wine_data(
    path: Path = RES_DIR / "ex6Data.csv",
) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]

    data = np.asarray(rows, dtype=float)
    if header[-1] != "quality" or data.shape[1] != 12:
        raise ValueError("expected 11 feature columns and one quality label column")
    return data[:, :-1], data[:, -1].astype(int)


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int = 10, seed: int = RANDOM_STATE
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)
        for fold_index, part in enumerate(np.array_split(label_indices, n_splits)):
            folds[fold_index].extend(part.tolist())
    return [np.array(sorted(fold), dtype=int) for fold in folds]


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    criterion: str,
    max_depth: int | None,
    min_samples_leaf: int,
    min_samples_split: int = 12,
    n_splits: int = 10,
) -> np.ndarray:
    indices = np.arange(y.size)
    scores = []
    for test_index in stratified_kfold_indices(y, n_splits=n_splits):
        train_index = np.setdiff1d(indices, test_index, assume_unique=True)
        model = DecisionTreeClassifier(
            criterion=criterion,  # type: ignore[arg-type]
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        )
        model.fit(x[train_index], y[train_index])
        predictions = model.predict(x[test_index])
        scores.append(float(np.mean(predictions == y[test_index])))
    return np.asarray(scores)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def precision_recall_f1(matrix: np.ndarray) -> tuple[float, float, float]:
    tp = matrix[1, 1]
    fp = matrix[0, 1]
    fn = matrix[1, 0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def export_tree_text(
    node: TreeNode,
    output_path: Path,
    feature_names: list[str],
    max_depth: int = 5,
) -> str:
    lines: list[str] = []

    def walk(current: TreeNode, depth: int, prefix: str) -> None:
        indent = "  " * depth
        summary = (
            f"samples={current.samples}, pos={current.positives}, "
            f"p1={current.probability:.3f}, predict={current.prediction}"
        )
        if current.is_leaf or depth >= max_depth:
            lines.append(f"{indent}{prefix}Leaf({summary})")
            return
        assert current.feature_index is not None and current.threshold is not None
        feature = feature_names[current.feature_index]
        lines.append(
            f"{indent}{prefix}{feature} <= {current.threshold:.5g} "
            f"(gain={current.gain:.4f}, {summary})"
        )
        assert current.left is not None and current.right is not None
        walk(current.left, depth + 1, "T: ")
        walk(current.right, depth + 1, "F: ")

    walk(node, 0, "")
    text = "\n".join(lines)
    output_path.write_text(text + "\n", encoding="utf-8")
    return text


def collect_plot_nodes(
    node: TreeNode,
    depth: int,
    max_depth: int,
    feature_names: list[str],
    nodes: list[tuple[TreeNode, int, float, str]],
    edges: list[tuple[float, int, float, int, str]],
    x_min: float,
    x_max: float,
    edge_label: str = "",
) -> float:
    x_center = (x_min + x_max) / 2.0
    if not node.is_leaf and depth < max_depth:
        assert node.left is not None and node.right is not None
        left_x = collect_plot_nodes(
            node.left,
            depth + 1,
            max_depth,
            feature_names,
            nodes,
            edges,
            x_min,
            x_center,
            "yes",
        )
        right_x = collect_plot_nodes(
            node.right,
            depth + 1,
            max_depth,
            feature_names,
            nodes,
            edges,
            x_center,
            x_max,
            "no",
        )
        edges.append((x_center, depth, left_x, depth + 1, "yes"))
        edges.append((x_center, depth, right_x, depth + 1, "no"))

    if node.is_leaf or depth >= max_depth:
        label = (
            f"Leaf: {node.prediction}\np(1)={node.probability:.2f}\nn={node.samples}"
        )
    else:
        assert node.feature_index is not None and node.threshold is not None
        label = (
            f"{feature_names[node.feature_index]}\n"
            f"<= {node.threshold:.3g}\n"
            f"gain={node.gain:.3f}"
        )
    nodes.append((node, depth, x_center, label))
    return x_center


def plot_tree(node: TreeNode, feature_names: list[str], output_path: Path) -> None:
    nodes: list[tuple[TreeNode, int, float, str]] = []
    edges: list[tuple[float, int, float, int, str]] = []
    max_depth = 3
    collect_plot_nodes(node, 0, max_depth, feature_names, nodes, edges, 0.0, 1.0)

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    for x1, y1, x2, y2, label in edges:
        ax.plot([x1, x2], [-y1, -y2], color="#6B7280", linewidth=1.4)
        ax.text((x1 + x2) / 2, -(y1 + y2) / 2, label, fontsize=9, color="#374151")
    for tree_node, depth, x, label in nodes:
        facecolor = "#E8F5E9" if tree_node.is_leaf or depth >= max_depth else "#E3F2FD"
        ax.text(
            x,
            -depth,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": facecolor,
                "edgecolor": "#374151",
                "linewidth": 1,
            },
        )
    ax.set_axis_off()
    ax.set_title("Decision Tree Structure (Top 3 Levels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_class_distribution(y: np.ndarray, output_path: Path) -> None:
    counts = np.bincount(y, minlength=2)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        ["0: quality <= 6", "1: quality >= 7"], counts, color=["#4C78A8", "#F58518"]
    )
    for bar, count in zip(bars, counts, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    plt.ylabel("Samples")
    plt.title("Wine Quality Binary Label Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_cv_scores(results: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(row["name"]) for row in results]
    means = [float(row["mean_accuracy"]) for row in results]
    stds = [float(row["std_accuracy"]) for row in results]
    order = np.argsort(means)
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    plt.figure(figsize=(10, 5))
    plt.barh(labels, means, xerr=stds, color="#59A14F", alpha=0.85)
    plt.axvline(0.78, color="#E15759", linestyle="--", linewidth=1, label="0.78 target")
    plt.xlabel("10-fold mean accuracy")
    plt.title("Decision Tree Cross Validation Accuracy")
    plt.xlim(min(0.68, min(means) - 0.02), 0.86)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, y = load_wine_data()
    plot_class_distribution(y, OUTPUT_DIR / "class_distribution.png")

    configs = [
        ("gini-depth3-leaf6-split12", "gini", 3, 6, 12),
        ("gini-depth5-leaf6-split12", "gini", 5, 6, 12),
        ("gini-depth9-leaf6-split12", "gini", 9, 6, 12),
        ("entropy-depth5-leaf6-split12", "entropy", 5, 6, 12),
        ("entropy-depth9-leaf6-split12", "entropy", 9, 6, 12),
        ("entropy-full-leaf6-split12", "entropy", None, 6, 12),
        ("entropy-full-leaf4-split12", "entropy", None, 4, 12),
        ("entropy-full-leaf3-split6", "entropy", None, 3, 6),
        ("entropy-full-leaf2-split2", "entropy", None, 2, 2),
        ("entropy-full-leaf1-split4", "entropy", None, 1, 4),
        ("entropy-full-leaf1-split2", "entropy", None, 1, 2),
    ]

    results: list[dict[str, object]] = []
    print("=== Dataset ===")
    print(f"samples={x.shape[0]}, features={x.shape[1]}")
    print(f"class 0={int((y == 0).sum())}, class 1={int((y == 1).sum())}")
    print()
    print("=== 10-fold Cross Validation ===")
    for name, criterion, max_depth, min_leaf, min_split in configs:
        scores = cross_validate(x, y, criterion, max_depth, min_leaf, min_split)
        row: dict[str, object] = {
            "name": name,
            "criterion": criterion,
            "max_depth": "None" if max_depth is None else max_depth,
            "min_samples_leaf": min_leaf,
            "min_samples_split": min_split,
            "mean_accuracy": f"{scores.mean():.6f}",
            "std_accuracy": f"{scores.std(ddof=1):.6f}",
            "fold_scores": " ".join(f"{score:.4f}" for score in scores),
        }
        results.append(row)
        print(
            f"{name:30s} mean={scores.mean():.4f}, std={scores.std(ddof=1):.4f}, "
            f"folds={np.array2string(scores, precision=4)}"
        )

    save_csv(
        OUTPUT_DIR / "cv_results.csv",
        results,
        [
            "name",
            "criterion",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "mean_accuracy",
            "std_accuracy",
            "fold_scores",
        ],
    )
    plot_cv_scores(results, OUTPUT_DIR / "cv_accuracy.png")

    best = max(results, key=lambda row: float(row["mean_accuracy"]))
    max_depth_value = None if best["max_depth"] == "None" else int(best["max_depth"])
    final_model = DecisionTreeClassifier(
        criterion=str(best["criterion"]),  # type: ignore[arg-type]
        max_depth=max_depth_value,
        min_samples_leaf=int(best["min_samples_leaf"]),
        min_samples_split=int(best["min_samples_split"]),
    )
    final_model.fit(x, y)
    predictions = final_model.predict(x)
    matrix = confusion_matrix(y, predictions)
    precision, recall, f1 = precision_recall_f1(matrix)

    np.savetxt(OUTPUT_DIR / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    assert final_model.root is not None
    tree_text = export_tree_text(
        final_model.root,
        OUTPUT_DIR / "tree_structure.txt",
        FEATURE_NAMES,
        max_depth=5,
    )
    plot_tree(final_model.root, FEATURE_NAMES, OUTPUT_DIR / "tree_top_levels.png")

    print()
    print("=== Best Model ===")
    print(
        f"name={best['name']}, criterion={best['criterion']}, "
        f"max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']}, "
        f"min_samples_split={best['min_samples_split']}"
    )
    print(f"mean_accuracy={float(best['mean_accuracy']):.4f}")
    print("training confusion matrix:")
    print(matrix)
    print(f"training precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    print()
    print("=== Tree Preview ===")
    print("\n".join(tree_text.splitlines()[:18]))
    print()
    print(f"outputs saved to: {OUTPUT_DIR}")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
