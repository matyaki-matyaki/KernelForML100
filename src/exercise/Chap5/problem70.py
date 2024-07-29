"""問題70"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sigma = 1
def k(x: np.ndarray, y: np.ndarray) -> float:
    """ガウシアンカーネル"""
    return np.exp(-(x - y)**2 / (2 * sigma**2))

# データの生成
m = 150
n = 100
rs = np.random.RandomState(42)
xx = rs.randn(m)
yy = rs.randn(n)

# 帰無分布の計算
T: list[float] = []
for _ in range(100):
    index1 = rs.choice(m, size=int(m/2), replace=False)
    index2 = rs.choice(n, size=int(n/2), replace=False)
    x_perm = np.concatenate((xx[index1], yy[index2]))
    y_perm = np.concatenate((xx[np.setdiff1d(np.arange(m), index1)], yy[np.setdiff1d(np.arange(n), index2)]))
    m_x_perm = x_perm.shape[0]
    n_y_perm = y_perm.shape[0]
    # (5.4)の計算
    term1 = sum(k(x_perm[i], x_perm[j]) for i in range(m_x_perm) for j in range(m_x_perm) if i != j) / (m_x_perm * (m_x_perm - 1))
    term2 = sum(k(y_perm[i], y_perm[j]) for i in range(n_y_perm) for j in range(n_y_perm) if i != j) / (n_y_perm * (n_y_perm - 1))
    term3 = sum(k(x_perm[i], y_perm[j]) for i in range(m_x_perm) for j in range(n_y_perm)) * 2 / (m_x_perm * n_y_perm)
    T.append(term1 + term2 - term3)
v = np.quantile(T, 0.95)

# 統計量の計算
term1 = sum(k(xx[i], xx[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
term2 = sum(k(yy[i], yy[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
term3 = sum(k(xx[i], yy[j]) for i in range(m) for j in range(n)) * 2 / (m * n)
u = term1 + term2 - term3

# グラフの図示
x = np.linspace(min(*T, u, v), max(*T, u, v), 200)
density = gaussian_kde(T)
plt.plot(x, density(x))
plt.axvline(x=u, c="r", linestyle="--")
plt.axvline(x=v, c="b")

filename = Path("src/exercise/Chap5/out/problem70/result.png")
filename.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(filename)
