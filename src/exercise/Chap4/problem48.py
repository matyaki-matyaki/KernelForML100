"""問題48"""
from typing import Callable

import numpy as np


def kernel_pca_train(X: np.ndarray, k: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """(4.9)のように中心化されたGram行列に対して最適なalphaを求める

    Args:
        X (np.ndarray): データ(N, p)
        k (Callable[[np.ndarray, np.ndarray], float]): カーネル

    Returns:
        np.ndarray: (4.9)のように中心化されたGram行列に対して最適なalpha(N, N)
    """
    # データXとカーネルkからGram行列を求める
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = k(X[i], X[j])
    # (4.9)に従って中心化
    S = [np.sum(K[i]) for i in range(N)]
    T = [np.sum(K[:, j]) for j in range(N)]
    U = np.sum(K)
    for i in range(N):
        for j in range(N):
            K[i, j] = K[i, j] - S[i] / N - T[j] / N + U / N**2
    # 固有値と固有ベクトルを用いて最適なalphaを求める
    val, vec = np.linalg.eigh(K)
    idX = val.argsort()[::-1]
    val = val[idX]
    vec = vec[idX]
    alpha = np.zeros_like(K)
    for i in range(N):
        alpha[:, i] = vec[:, i] / np.sqrt(np.maximum(val[i], 1e-6))
    return alpha

def kernel_pca_test(
    X: np.ndarray,
    k: Callable[[np.ndarray, np.ndarray], float],
    alpha: np.ndarray,
    m: int,
    z: np.ndarray,
) -> np.ndarray:
    """X, k, alpha(kernel_pca_train(X, k)), m, zからm次までのスコアpcaを求める

    Args:
        X (np.ndarray): データ(N, p)
        k (Callable[[np.ndarray, np.ndarray], float]): カーネル
        alpha (np.ndarray): kernel_pca_train(X, k)
        m (int): 1<= m <= p
        z (np.ndarray): Xのある行x_i (i in [N])(p)

    Returns:
        np.ndarray: X, k, alpha(kernel_pca_train(X, k)), m, zからm次までのスコアpca
    """
    N = X.shape[0]
    pca = np.zeros(m)
    for i in range(N):
        pca = pca + alpha[i, 0:m] * k(X[i], z)
    return pca

if __name__ == "__main__":
    """動作確認"""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    sigma2 = 1e-2
    def k(x: np.ndarray, y: np.ndarray) -> float:
        """Gaussian kernel."""
        return np.exp(-np.linalg.norm(x - y)**2 / 2 / sigma2)
    X = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/USArrests.csv")
    x = X.to_numpy()[:, :-1]
    n = x.shape[0]
    p = x.shape[1]
    alpha = kernel_pca_train(x, k)
    z = np.zeros((n, 2))
    for i in range(n):
        z[i, :] = kernel_pca_test(x, k, alpha, 2, x[i, :])

    min1 = np.min(z[:, 0])
    min2 = np.min(z[:, 1])
    max1 = np.max(z[:, 0])
    max2 = np.max(z[:, 1])
    plt.xlim(min1, max1)
    plt.ylim(min2, max2)
    plt.xlabel("First")
    plt.ylabel("Second")
    plt.title("Kernel PCA (Gauss 0.01)")
    for i in range(n):
        if i != 4:  # noqa: PLR2004
            plt.text(x=z[i, 0], y=z[i, 1], s=i)  # type:ignore[arg-type]
    plt.text(z[4, 0], z[4, 1], 5, c="r")  # type:ignore[arg-type]
    filename = Path("src/exercise/Chap4/out/problem48/result.png")
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)
    plt.close()
