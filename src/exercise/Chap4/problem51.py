"""問題51"""
from typing import Callable

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix


def K_linear(x: np.ndarray, y: np.ndarray) -> float:
    """線形カーネル"""
    return x.T @ y


def K_poly(x: np.ndarray, y: np.ndarray) -> float:
    """多項式カーネル"""
    return (1 + x.T @ y)**2

def create_K_Gaussian(sigma2: float) -> Callable[[np.ndarray, np.ndarray], float]:
    """ガウシアンカーネルのグラム行列を返す関数を作成"""
    def K_Gaussian(x: np.ndarray, y: np.ndarray) -> float:
        return np.exp(-np.linalg.norm(x - y)**2 / 2 / sigma2)
    return K_Gaussian

def svm_2(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    K: Callable[[np.ndarray, np.ndarray], float],
)-> dict[str, np.ndarray]:
    """SVMの最適解を出力"""
    eps = 0.0001
    n = X.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = K(X[i, :], X[j, :]) * y[i] * y[j]
    # パッケージにある matrix 関数を使って指定する必要がある
    P = matrix(P + np.eye(n) * eps)
    A = matrix(-y.T.astype(float))
    b = matrix(np.array([0]).astype(float))
    h = matrix(np.array([C] * n + [0] * n).reshape(-1, 1).astype(float))
    G = matrix(np.concatenate([np.diag(np.ones(n)), np.diag(-np.ones(n))]))
    q = matrix(np.array([-1] * n).astype(float))
    res = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h)
    alpha = np.array(res["x"])  # x が本文中の alpha に対応
    beta = ((alpha * y).T @ X).reshape(2, 1)
    index = (eps < alpha[:, 0]) & (alpha[:, 0] < C - eps)
    beta_0 = np.mean(y[index] - X[index, :] @ beta)
    return {"alpha": alpha, "beta": beta, "beta_0": beta_0}


def plot_kernel(X: np.ndarray, y: np.ndarray, K: Callable[[np.ndarray, np.ndarray], float], line: str) -> None:
    """境界線を図示"""
    res = svm_2(X, y, 1, K)
    alpha = res["alpha"][:, 0]
    beta_0 = res["beta_0"]

    def f(u: float, v: float) -> float:
        """Function f."""
        S = beta_0
        for i in range(X.shape[0]):
            S = S + alpha[i] * y[i] * K(X[i, :], np.array([u, v]))
        return S[0]
    # ww は f(x,y) での高さ。これから輪郭を求めることができる
    uu = np.arange(-2, 2, 0.1)
    vv = np.arange(-2, 2, 0.1)
    ww = []
    for v in vv:
        w = [f(u, v) for u in uu]
        ww.append(w)
    plt.contour(uu, vv, ww, levels=0, linestyles=line)

if __name__ == "__main__":
    """動作確認"""
    from pathlib import Path
    rs = np.random.RandomState(42)  # 乱数生成器を固定
    a = 3
    b = -1
    n = 200
    X = rs.randn(n, 2)
    y = np.sign(a * X[:, 0] + b * X[:, 1]**2 + 0.3 * rs.randn(n))
    y = y.reshape(-1, 1)
    for i in range(n):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c="red")
        else:
            plt.scatter(X[i, 0], X[i, 1], c="blue")
    plot_kernel(X, y, create_K_Gaussian(sigma2=1e0), line="dashdot")
    plot_kernel(X, y, create_K_Gaussian(sigma2=1e2), line="dashed")
    plot_kernel(X, y, create_K_Gaussian(sigma2=1e4), line="solid")
    plt.title("sigma2 is 1e0 (dashdot), 1e2 (dashed), 1e4 (solid)")

    filename = Path("src/exercise/Chap4/out/problem51/result.png")
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)
    plt.close()
