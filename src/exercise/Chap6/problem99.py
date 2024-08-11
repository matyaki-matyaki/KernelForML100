"""問題99"""
import numpy as np
import skfda
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

X, y = skfda.datasets.fetch_weather(return_X_y=True, as_frame=True)
df = X.iloc[:, 0].values


def g(j, x):                 # 基底を p 個用意する
    if j == 0:
        return 1 / np.sqrt(2 * np.pi)
    if j % 1 == 0:
        return np.cos((j // 2) * x) / np.sqrt(np.pi)
    else:
        return np.sin((j // 2) * x) / np.sqrt(np.pi)


def beta(x, y):              # 関数の p 個の基底の係数を計算
    X = np.zeros((N, p))
    for i in range(N):
        for j in range(p):
            X[i, j] = g(j, x[i])
    beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)
                                       + 0.0001 * np.identity(p)), X.T), y)
    return np.squeeze(beta)


N = 365
n = 35
m = 5
p = 100

# df.coordinates[0]: 気温, df.coordinates[1]: 降水量:
df = df.coordinates[1].data_matrix
C = np.zeros((n, p))
for i in range(n):
    x = np.arange(1, N+1) * (2 * np.pi / N) - np.pi
    y = df[i]
    C[i, :] = beta(x, y)
pca = PCA()
pca.fit(C)
B = pca.components_.T
xx = C.dot(B)

x_seq = np.arange(-np.pi, np.pi, 2 * np.pi / 100)
colors = ["red", "green", "blue"]
colormap = plt.cm.gist_ncar  # nipy_spectral, Set1, Paired
color = [colormap(i) for i in np.linspace(0, 1, len(range(10)))]

def h(coef, x):    # 係数を用いて関数を定義
    S = 0
    for j in range(p):
        S = S + coef[j] * g(j, x)
    return S

plt.figure()
plt.xlim(-np.pi, np.pi)
plt.ylim(-1, 1)
for j in range(3):
    plt.plot(x_seq, h(B[:, j], x_seq), c=colors[j], label="PC%d" % (j+1))
plt.legend(loc="best")
plt.show()

place = X.iloc[:, 1]
index = [9, 11, 12, 13, 16, 23, 25, 26]
others = [x for x in range(34) if x not in index]
first = [place[i][0] for i in index]

plt.figure()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xlim(min(xx[:, 0]) * 1.1, max(xx[:, 0]) * 1.1)
plt.ylim(min(xx[:, 1]) * 1.1, max(xx[:, 1]) * 1.1)
plt.title("Canadian Weather")
plt.scatter(xx[others, 0], xx[others, 1], marker="x", c="k")
for i in range(8):
    l = plt.text(xx[index[i], 0], xx[index[i], 1], s=first[i], c=color[i])
plt.show()
