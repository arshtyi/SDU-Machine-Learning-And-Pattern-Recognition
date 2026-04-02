from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

data_dir = Path("res")
x_raw = np.loadtxt(data_dir / "ex4x.dat")
y = np.loadtxt(data_dir / "ex4y.dat")

m = x_raw.shape[0]
x = np.c_[np.ones(m), x_raw]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def cost(theta: np.ndarray, x_mat: np.ndarray, y_vec: np.ndarray) -> float:
    h = sigmoid(x_mat @ theta)
    eps = 1e-12
    return float(-np.mean(y_vec * np.log(h + eps) + (1 - y_vec) * np.log(1 - h + eps)))


theta = np.zeros(x.shape[1], dtype=float)
max_iter = 30
tol = 1e-8
cost_history: list[float] = []

for _ in range(max_iter):
    h = sigmoid(x @ theta)
    grad = (x.T @ (h - y)) / m
    w = h * (1 - h)
    hessian = (x.T @ (x * w[:, None])) / m
    step = np.linalg.solve(hessian, grad)
    theta = theta - step
    cost_history.append(cost(theta, x, y))
    if np.linalg.norm(step, ord=2) < tol:
        break

iters_used = len(cost_history)

pos = y == 1
neg = y == 0

plt.figure(figsize=(6, 5))
plt.scatter(x_raw[pos, 0], x_raw[pos, 1], marker="+", s=80, label="Admitted")
plt.scatter(x_raw[neg, 0], x_raw[neg, 1], marker="o", s=35, label="Not admitted")
plt.xlabel("Exam 1")
plt.ylabel("Exam 2")
plt.title("Training Data")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show(block=True)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, iters_used + 1), cost_history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("J(theta)")
plt.title("Newton Method Convergence")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show(block=True)
plt.close()

x1_line = np.linspace(x_raw[:, 0].min() - 2, x_raw[:, 0].max() + 2, 200)
if abs(theta[2]) < 1e-12:
    x2_line = np.full_like(x1_line, np.nan)
else:
    x2_line = -(theta[0] + theta[1] * x1_line) / theta[2]

plt.figure(figsize=(6, 5))
plt.scatter(x_raw[pos, 0], x_raw[pos, 1], marker="+", s=80, label="Admitted")
plt.scatter(x_raw[neg, 0], x_raw[neg, 1], marker="o", s=35, label="Not admitted")
plt.plot(x1_line, x2_line, "r-", linewidth=2, label="Decision boundary")
plt.xlabel("Exam 1")
plt.ylabel("Exam 2")
plt.title("Logistic Regression with Newton Method")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show(block=True)
plt.close()

exam = np.array([1.0, 20.0, 80.0])
p_admit = float(sigmoid(exam @ theta))
p_not_admit = 1.0 - p_admit

print("theta =", np.array2string(theta, precision=6, suppress_small=True))
print("iterations =", iters_used)
print("P(not admitted | Exam1=20, Exam2=80) =", f"{p_not_admit:.6f}")
