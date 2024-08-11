"""問題95"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

colormap = plt.cm.gist_ncar  # nipy_spectral, Set1, Paired
colors = [colormap(i) for i in np.linspace(0, 0.8, 5)]

def matern(nu, l, r):
    p = nu - 1 / 2
    S = 0
    for i in range(int(p+1)):
        S = S + gamma(p + i + 1) / gamma(i + 1) / gamma(p - i + 1) \
            * (np.sqrt(8 * nu) * r / l)**(p - i)
    S = S * gamma(p + 2) / gamma(2 * p + 1) * np.exp(-np.sqrt(2 * nu) * r / l)
    return S

def rand_100(Sigma):
    L = np.linalg.cholesky(Sigma)    # 共分散行列を Cholesky 分解
    u = np.random.randn(100)
    y = L.dot(u)  # 平均 0 の共分散行列 Sigma の乱数を 1 組発生
    return y


x = np.linspace(0, 1, 100)
z = np.abs(np.subtract.outer(x, x))  # 距離行列を計算
l = 0.1
nu = 5 / 2

plt.figure()
Sigma_M = matern(nu, l, z)          # 行列
y = rand_100(Sigma_M)
plt.plot(x, y)
plt.ylim(-3, 3)
for i in range(5):
    y = rand_100(Sigma_M)
    plt.plot(x, y, c=colors[i])
plt.title(f"Matern process ({nu=}, {l=})")

filename = Path("src/exercise/Chap6/out/problem95/result.png")
filename.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(filename)
