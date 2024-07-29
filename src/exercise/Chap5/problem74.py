"""問題74"""
import numpy as np


def HSIC(x, y, k_x, k_y):
    n = len(x)
    S = 0
    for i in range(n):
        for j in range(n):
            S = S + k_x(x[i], x[j]) * k_y(y[i], y[j])
    T = 0
    for i in range(n):
        T_1 = 0
        for j in range(n):
            T_1 = T_1 + k_x(x[i], x[j])
        T_2 = 0
        for l in range(n):
            T_2 = T_2 + k_y(y[i], y[l])
        T = T + T_1 * T_2
    U = 0
    for i in range(n):
        for j in range(n):
            U = U + k_x(x[i], x[j])
    V = 0
    for i in range(n):
        for j in range(n):
            V = V + k_y(y[i], y[j])
    return S / n**2 - 2 * T / n**3 + U * V / n**4


def HSIC_trace(x, y, k_x, k_y):
    n = len(x)
    K_x = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_x[i, j] = k_x(x[i], x[j])
    K_y = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_y[i, j] = k_y(y[i], y[j])
    E = np.ones((n, n))
    H = np.identity(n) - E / n
    return np.sum(np.diag(np.diag(K_x.dot(H).dot(K_y).dot(H)))) / n**2

if __name__ == "__main__":
    def k_x(x, y):
        return np. exp(-np.linalg.norm(x - y)**2 / 2)

    k_y = k_x
    k_z = k_x

    n = 100
    rs = np.random.RandomState(42)
    for a in [0, 0.1, 0.2, 0.4, 0.6, 0.8]:
        x = rs.randn(n)
        z = rs.randn(n)
        y = a * x + np.sqrt(1 - a**2) * z
        print(f"{a=}, {HSIC(x, y, k_x, k_y)=}, {HSIC_trace(x, y, k_x, k_y)}")
    """
    a=0, HSIC(x, y, k_x, k_y)=0.0016144347904881728, 0.0016144347904855324
    a=0.1, HSIC(x, y, k_x, k_y)=0.0024845938768538467, 0.002484593876853815
    a=0.2, HSIC(x, y, k_x, k_y)=0.0020244572354856105, 0.0020244572354858425
    a=0.4, HSIC(x, y, k_x, k_y)=0.0032241047523186017, 0.003224104752316417
    a=0.6, HSIC(x, y, k_x, k_y)=0.011470114073025228, 0.011470114073027894
    a=0.8, HSIC(x, y, k_x, k_y)=0.019901742290715563, 0.019901742290714505
    """
