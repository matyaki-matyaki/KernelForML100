"""問題42（Fを出力するとき、入力iは不要であると判断した。）"""
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def eigen_functions_from_kernel(
    x: np.ndarray,
    k: Callable[[np.ndarray, np.ndarray], float],
) -> Callable[[np.ndarray, int], float]:
    """例62のプログラムの一部"""
    m = x.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = k(x[i], x[j])
    values, vectors = np.linalg.eigh(K)
    alpha = np.zeros((m, m))
    for i in range(m):
        alpha[:, i] = vectors[:, i] * np.sqrt(m) / (values[i] + 1e-16)

    def F(y: np.ndarray, i: int) -> float:
        """Function F."""
        S = 0
        for j in range(m):
            S += alpha[j, i] * k(x[i], y)
        return S

    return F

if __name__ == "__main__":
    from pathlib import Path
    # カーネルの定義
    sigma = 1
    def k(x: np.ndarray, y: np.ndarray) -> float:
        """Gaussian kernel."""
        return np.exp(-(x - y)**2 / sigma**2)

    # サンプルの発生
    m = 300
    rs = np.random.RandomState(42)
    x = rs.randn(m) - 2 * rs.randn(m)**2 + 3 * rs.randn(m)**3

    # 実行
    eig_functions = eigen_functions_from_kernel(x, k)
    w = np.linspace(-2, 2, 100)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        ax.plot(w, eig_functions(w, i))
        ax.set_title(f"{i+1} th Eigenfunction")
    plt.suptitle("Eigen Functions")
    out_dir = Path("src/exercise/Chap3/out/problem42")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(out_dir, "eigenfunctions.png"))
    plt.close()
