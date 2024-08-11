"""問題93"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def lam(j):           # 固有値
    return 4 / ((2 * j - 1) * np.pi)**2

def ee(j, x):         # 固有関数の定義
    return np.sqrt(2) * np.sin((2 * j - 1) * np.pi / 2 * x)

n = 10
m = 5

# Gauss 過程の定義
def f(z, x):
    n = len(z)
    S = 0
    for i in range(n):
        S = S + z[i] * ee(i, x) * np.sqrt(lam(i))
    return S

plt.figure()
plt.xlim(0, 1)
plt.xlabel("x")
plt.ylabel("f(omega, x)")
colormap = plt.cm.gist_ncar  # nipy_spectral, Set1, Paired
colors = [colormap(i) for i in np.linspace(0, 0.8, m)]

for j in range(m):
    z = np.random.randn(n)
    x_seq = np.arange(0, 3.001, 0.001)
    y_seq = []
    for x in x_seq:
        y_seq.append(f(z, x))
    plt.plot(x_seq, y_seq, c=colors[j])

plt.title("Brown Motion")
filename = Path("src/exercise/Chap6/out/problem93/result.png")
filename.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(filename)
