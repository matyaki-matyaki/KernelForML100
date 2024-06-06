"""問題2の前半（D, fの動作の確認(Epanechnikov)）"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def D(t: float) -> float:
    """Function D for Epanechnikov kernel."""
    if np.abs(t) <= 1.:
        return 3 * (1 - t ** 2) / 4
    return 0.

def k(x: float, y: float, lam: float) -> float:
    """Epanechnikov kernel."""
    return D(np.abs((x - y) / lam))


def f(observed_data: list[tuple[float, float]], x: float, lam: float) -> float:
    """Nadaraya-Watson Estimator."""
    numerator = np.sum([k(x, x_i, lam) * y_i for x_i, y_i in observed_data])
    denominator = np.sum([k(x, x_i, lam) for x_i, _ in observed_data])
    if denominator == 0:
        return 0.
    return numerator / denominator


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
lam = [0.05, 0.35, 0.50]
color = ["g", "b", "r"]
for i in range(3):
    for zz in xx:
        yy[i].append(f(observed_data, zz, lam[i]))
    plt.plot(xx, yy[i], c=color[i], label=lam[i])

plt.legend(loc="upper left", frameon=True, prop={"size": 14})
plt.title("Nadaraya-Watson Estimator", fontsize=20)

# Save.
out_dir = Path("src/exercise/Chap1/out/problem2")
out_dir.mkdir(parents=True, exist_ok=True)
filename = Path(out_dir, "Epanechnikov.png")
plt.savefig(filename)
