"""問題54"""
import numpy as np

m = 100
sigma = 10
sigma2 = sigma ** 2
rs = np.random.RandomState(42)

def k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gaussian kernel."""
    return np.exp(-(x - y) ** 2 / (2 * sigma2))

def z(x: np.ndarray, w:np.ndarray, b: np.ndarray) -> np.ndarray:
    """Function z."""
    return np.sqrt(2 / m) * np.cos(w * x + b)

def zz(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    """Function zz."""
    return np.sum(z(x, w, b) * z(y, w, b))

def estimate_hat_k(x: np.ndarray, y: np.ndarray) -> float:
    """hat_kを推定"""
    w = rs.randn(m) / sigma
    b = rs.rand(m) * 2 * np.pi
    return zz(x, y, w, b)

ITER = 1000
gap = np.zeros(ITER)
for i in range(ITER):
    x = rs.randn(1)
    y = rs.randn(1)
    gap_i = np.abs(estimate_hat_k(x, y) - k(x, y))
    gap[i] = gap_i

print("Mean of gap (absolute error) is ", np.mean(gap))
