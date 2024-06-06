"""問題8"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def K(x: float, y: float, sigma2: float) -> float:
    """Gaussian Kernel."""
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma2))

rs = np.random.RandomState(42)
n = 100
x = 2 * rs.normal(size=n)
y = np.sin(2 * np.pi * x) + rs.normal(size=n) / 4

# 最適なsigma2の探索
sigma2_seq = np.arange(1e-3, 1e-2, 1e-3)
SS_min = np.inf
for sigma2 in sigma2_seq:
    SS = 0
    for k in range(n):
        test = [k]
        train = [x for x in range(n) if x not in test]
        for j in test:
            u, v = 0., 0.
            for i in train:
                kk = K(x[i], x[j], sigma2)
                u = u + kk * y[i]
                v = v + kk
            if v != 0:
                z = u / v
                SS = SS + (y[j] - z) ** 2
    if SS_min > SS:
        SS_min = SS
        sigma2_best = sigma2

print("Best sigma2 =", sigma2_best)

# 最適なsigma2を用いて曲線を表示
def f(observed_data: list[tuple[float, float]], x: float, sigma2: float) -> float:
    """Nadaraya-Watson Estimator."""
    numerator = np.sum([K(x, x_i, sigma2) * y_i for x_i, y_i in observed_data])
    denominator = np.sum([K(x, x_i, sigma2) for x_i, _ in observed_data])
    if denominator == 0:
        return 0.
    return numerator / denominator

xx = np.arange(-3, 3, 0.1)
observed_data = [(x[i], y[i]) for i in range(n)]
yy = [f(observed_data, zz, sigma2_best) for zz in xx]

# 表示
plt.figure(num=1, figsize=(15, 8), dpi=80)
plt.xlim(-3, 3)
plt.ylim(-2, 3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.scatter(x, y, facecolors="none", edgecolors="k", marker="o")
plt.plot(xx, yy, c="r")
plt.title(f"Best sigma2 = {sigma2_best}")

# Save.
out_dir = Path("src/exercise/Chap1/out/problem8")
out_dir.mkdir(parents=True, exist_ok=True)
filename = Path(out_dir, "optimal_sigma2.png")
plt.savefig(filename)
