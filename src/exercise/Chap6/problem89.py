"""問題89"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

df = load_iris()           # Iris データ
x = df.data[50:150, 0:4]
y = np.array([1]*50 + [-1]*50)
n = len(y)


# 4 個の共変量でカーネルを計算
def k(x, y):
    return np.exp(np.sum(-(x - y)**2) / 2)


K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K[i, j] = k(x[i, :], x[j, :])
eps = 0.00001
f = [0] * n
g = [0.1] * n

while np.sum((np.array(f) - np.array(g))**2) > eps:
    i = i + 1
    g = f                 # 比較のため，更新前の値を保存する
    f = np.array(f)
    y = np.array(y)
    v = np.exp(-y * f)
    u = y * v / (1 + v)
    w = (v / (1 + v)**2)
    W = np.diag(w)
    W_p = np.diag(w**0.5)
    W_m = np.diag(w**(-0.5))
    L = np.linalg.cholesky(np.identity(n) + np.dot(np.dot(W_p, K), W_p))
    gamma = W.dot(f) + u
    beta = np.linalg.solve(L, np.dot(np.dot(W_p, K), gamma))
    alpha = np.linalg.solve(np.dot(L.T, W_m), beta)
    f = np.dot(K, (gamma - alpha))
print(list(f))

plt.scatter(range(len(f)), list(f))
filename = Path("src/exercise/Chap6/out/problem89/result.png")
filename.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(filename)
