from pathlib import Path
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "res"


def load_points(name: str) -> np.ndarray:
    path = DATA_DIR / f"{name}.dat"
    return np.loadtxt(path)


def class_scatter(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    centered = points - mean
    scatter = centered.T @ centered
    return mean, scatter


def lda_two_classes(class_a: np.ndarray, class_b: np.ndarray):
    mean_a, scatter_a = class_scatter(class_a)
    mean_b, scatter_b = class_scatter(class_b)
    sw = scatter_a + scatter_b
    mean_diff = (mean_a - mean_b).reshape(-1, 1)
    w = np.linalg.solve(sw, mean_diff).ravel()
    w = w / np.linalg.norm(w)
    origin = (mean_a + mean_b) / 2
    return w, origin


def project_onto_line(points: np.ndarray, origin: np.ndarray, direction: np.ndarray):
    direction = direction / np.linalg.norm(direction)
    scalars = (points - origin) @ direction
    projections = origin + np.outer(scalars, direction)
    return projections, scalars


def plot_two_class_raw(red: np.ndarray, blue: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(red[:, 0], red[:, 1], c="red", label="Red class")
    ax.scatter(blue[:, 0], blue[:, 1], c="blue", label="Blue class")
    ax.set_title("Figure 1: Raw red & blue points")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()


def plot_two_class_line_and_projection(red: np.ndarray, blue: np.ndarray):
    w, origin = lda_two_classes(red, blue)
    t_range = np.linspace(-3, 3, 20)
    line_pts = origin + np.outer(t_range, w)
    proj_red, _ = project_onto_line(red, origin, w)
    proj_blue, _ = project_onto_line(blue, origin, w)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(red[:, 0], red[:, 1], c="red", label="Red class")
    ax2.scatter(blue[:, 0], blue[:, 1], c="blue", label="Blue class")
    ax2.plot(line_pts[:, 0], line_pts[:, 1], "k-", label="LDA line")
    ax2.set_title("Figure 2: LDA separating line (2 classes)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle=":", alpha=0.5)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.scatter(red[:, 0], red[:, 1], c="red", alpha=0.35, label="Red class")
    ax3.scatter(blue[:, 0], blue[:, 1], c="blue", alpha=0.35, label="Blue class")
    ax3.plot(line_pts[:, 0], line_pts[:, 1], "k-", label="LDA line")
    ax3.scatter(
        proj_red[:, 0], proj_red[:, 1], c="darkred", s=25, label="Red projections"
    )
    ax3.scatter(
        proj_blue[:, 0], proj_blue[:, 1], c="navy", s=25, label="Blue projections"
    )
    for p, q in zip(red, proj_red):
        ax3.plot([p[0], q[0]], [p[1], q[1]], "r--", linewidth=0.8, alpha=0.6)
    for p, q in zip(blue, proj_blue):
        ax3.plot([p[0], q[0]], [p[1], q[1]], "b--", linewidth=0.8, alpha=0.6)
    ax3.set_title("Figure 3: Projection of red/blue onto LDA line")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.legend(loc="best")
    ax3.grid(True, linestyle=":", alpha=0.5)
    fig3.tight_layout()


def lda_multi_class(classes: Iterable[np.ndarray], n_components: int = 2):
    classes = list(classes)
    d = classes[0].shape[1]
    overall_mean = np.vstack(classes).mean(axis=0)

    sw = np.zeros((d, d))
    sb = np.zeros((d, d))

    for cls in classes:
        mean_c, scatter_c = class_scatter(cls)
        sw += scatter_c
        diff = (mean_c - overall_mean).reshape(-1, 1)
        sb += cls.shape[0] * diff @ diff.T

    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(sw) @ sb)
    idx = np.argsort(eigvals.real)[::-1]
    eigvecs = eigvecs[:, idx]
    W = eigvecs[:, :n_components].real
    for i in range(W.shape[1]):
        W[:, i] /= np.linalg.norm(W[:, i]) + 1e-12
    return W, overall_mean


def plot_three_class_demo(red: np.ndarray, blue: np.ndarray, green: np.ndarray):
    classes = [red, blue, green]
    labels = ["Red", "Blue", "Green"]
    colors = ["red", "blue", "green"]

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    for data, label, color in zip(classes, labels, colors):
        ax1.scatter(data[:, 0], data[:, 1], c=color, label=label)
    ax1.set_title("Figure 4: Raw red/blue/green points")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=":", alpha=0.5)
    fig1.tight_layout()

    W, _ = lda_multi_class(classes, n_components=2)
    projected = [cls @ W for cls in classes]
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for data, label, color in zip(projected, labels, colors):
        ax2.scatter(data[:, 0], data[:, 1], c=color, label=label)
    ax2.set_title("3-Class LDA on first two discriminants")
    ax2.set_xlabel("LD1")
    ax2.set_ylabel("LD2")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle=":", alpha=0.5)
    fig2.tight_layout()


def main():
    red = load_points("ex3red")
    blue = load_points("ex3blue")
    green = load_points("ex3green")

    plot_two_class_raw(red, blue)
    plot_two_class_line_and_projection(red, blue)
    plot_three_class_demo(red, blue, green)

    plt.show()


if __name__ == "__main__":
    main()
