"""問題43"""
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def eigen_values_functions_from_kernel(
    x: np.ndarray,
    k: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[np.ndarray, Callable[[np.ndarray, int], float]]:
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

    return values, F

if __name__ == "__main__":
    from pathlib import Path
    # カーネルの定義
    def k(x: np.ndarray, y: np.ndarray) -> float:
        """Gaussian kernel."""
        return (1 + np.dot(x, y)) ** 2
    # サンプルの発生
    m = 1000
    rs = np.random.RandomState(42)
    x = rs.randn(m) - 2 * rs.randn(m)**2 + 3 * rs.randn(m)**3

    # 実行
    eigvalues, eigfunctions = eigen_values_functions_from_kernel(x, k)
    eigvalues = np.flip(eigvalues)
    # 固有値のプロット
    plt.plot(eigvalues[:100])
    plt.title("First 100 Eigenvalues")
    plt.xlabel("index")
    plt.ylabel("Eigenvalue")
    plt.yscale("log")
    out_dir = Path("src/exercise/Chap3/out/problem43")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(out_dir, "eigenvalues.png"))
    plt.close()
    w = np.linspace(-2, 2, 100)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        ax.plot(w, eigfunctions(w, i))
        ax.set_title(f"{i+1} th Eigenfunction")
    plt.suptitle("Eigen Functions")
    plt.savefig(Path(out_dir, "eigenfunctions.png"))
    plt.close()
