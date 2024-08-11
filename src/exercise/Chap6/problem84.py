"""問題84"""
import numpy as np


# (m, k) の定義
def m(x):
    return 0


def k(x, y):
    return np.exp(-(x-y)**2 / 2)


# 関数 gp_sample の定義
def gp_sample(x, m, k):
    n = len(x)
    m_x = m(x)
    k_xx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k_xx[i, j] = k(x[i], x[j])
    R = np.linalg.cholesky(k_xx)  # lower triangular matrix
    u = np.random.randn(n)
    return R.dot(u) + m_x


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    # 乱数を発生して，共分散行列を生成し，k_xx と比較
    x = np.arange(-2, 3, 1)
    n = len(x)
    r = 1000
    z = np.zeros((n, r))
    for i in range(r):
        z[:, i] = gp_sample(x, m, k)
    k_xx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k_xx[i, j] = k(x[i], x[j])

    print("cov(z):\n", np.cov(z), "\n")
    print("k_xx:\n", k_xx)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.cov(z))
    plt.title("cov")
    plt.subplot(1, 2, 2)
    plt.imshow(k_xx)
    plt.title("k_xx")

    filename = Path("src/exercise/Chap6/out/problem84/result.png")
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)
