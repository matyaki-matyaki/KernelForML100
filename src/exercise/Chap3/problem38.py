"""問題38"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def H(j: int, x: float) -> float:
    """Function H."""
    match j:
        case 0:
            return 1
        case 1:
            return 2 * x
        case 2:
            return -2 + 4 * x**2
        case _:
            return -12 * x + 8 * x**3

def phi(j: int, x: np.ndarray, sigma: float, sigma_hat: float) -> float:
    """Function phi."""
    a = (4 * sigma_hat**2) ** (-1)
    b = (2 * sigma**2) ** (-1)
    c = np.sqrt(a**2 + 2 * a * b)
    return np.exp(-(c - a) * x**2) * H(j, np.sqrt(2 * c) * x)

sigma_list = [1e-1, 1e0, 1e1]
sigma_hat_list = [1e-1, 1e0, 1e1]

out_dir = Path("src/exercise/Chap3/out/problem38")
out_dir.mkdir(parents=True, exist_ok=True)

x = np.linspace(-2, 2, 100)
color = ["b", "g", "r", "b"]

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i, sigma in enumerate(sigma_list):
    for ii, sigma_hat in enumerate(sigma_hat_list):
        ax = axs[i, ii]
        for j in range(4):
            ax.plot(x, phi(j, x, sigma, sigma_hat), c=color[j], label=f"{j=}")
        ax.set_ylabel("phi")
        ax.set_title(f"{sigma=}, {sigma_hat=}")
        ax.legend()

plt.suptitle("Characteristic function of Gauss Kernel.")
plt.savefig(Path(out_dir, "result.png"))
plt.close()
