"""問題2の後半（Epanechnikovカーネル以外）"""
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def f(observed_data: list[tuple[float, float]], x: float, k: Callable[[float, float], float]) -> float:
    """Nadaraya-Watson Estimator."""
    numerator = np.sum([k(x, x_i) * y_i for x_i, y_i in observed_data])
    denominator = np.sum([k(x, x_i) for x_i, _ in observed_data])
    if denominator == 0:
        return 0.
    return numerator / denominator

methods = ["Gaussian", "Exponential", "Polynomial"]

rs = np.random.RandomState(42)
n = 250
x = 2 * rs.normal(size=n)
y = np.sin(2 * np.pi * x) + rs.normal(size=n) / 4
observed_data = [(x[i], y[i]) for i in range(n)]

plt.figure(num=1, figsize=(15, 8), dpi=80)
plt.xlim(-3, 3)
plt.ylim(-2, 3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.scatter(x, y, facecolors="none", edgecolors="k", marker="o")

xx = np.arange(-3, 3, 0.1)
yy: list = [[] for _ in range(3)]
color = ["g", "b", "r"]

for i, method in enumerate(methods):
    match method:
        case "Gaussian":
            def k(x: float, y: float, sigma: float = 0.1) -> float:
                """Gaussian Kernel."""
                return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
        case "Exponential":
            def k(x: float, y: float, beta: float = 1.) -> float:
                """Exponential Kernel."""
                return np.exp(beta * np.dot(x, y))
        case "Polynomial":
            def k(x: float, y: float, m: int = 3) -> float:
                """Polynomial Kernel."""
                return (np.dot(x, y) + 1) ** m
        case _:
            msg = f"ValueError: Unknown method: {method}"

    for zz in xx:
        yy[i].append(f(observed_data, zz, k))
    plt.plot(xx, yy[i], c=color[i], label=method)

plt.legend(loc="upper left", frameon=True, prop={"size": 14})
plt.title("Nadaraya-Watson Estimator", fontsize=20)

# Save.
out_dir = Path("src/exercise/Chap1/out/problem2")
out_dir.mkdir(parents=True, exist_ok=True)
filename = Path(out_dir, "other_kernels.png")
plt.savefig(filename)
