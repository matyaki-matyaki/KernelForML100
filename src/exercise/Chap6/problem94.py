"""問題94"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma


def matern(nu, l, r):
    p = nu - 1 / 2
    S = 0
    for i in range(int(p+1)):
        S = S + gamma(p + i + 1) / gamma(i + 1) / gamma(p - i + 1) \
            * (np.sqrt(8 * nu) * r / l)**(p - i)
    S = S * gamma(p + 2) / gamma(2 * p + 1) * np.exp(-np.sqrt(2 * nu) * r / l)
    return S

if __name__ == "__main__":
    m = 10
    l = 0.05
    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1, Paired
    color = [colormap(i) for i in np.linspace(0, 1, len(range(m)))]
    x = np.linspace(0, 0.5, 200)
    plt.ylim(0, 10)
    for i in range(1, m + 1):
        plt.plot(x, matern(i, l, x), c=color[i - 1], label=r"$\nu=%d$" % i)

    plt.legend(loc="upper right", frameon=True, prop={"size": 14})
    filename = Path("src/exercise/Chap6/out/problem94/result.png")
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)
