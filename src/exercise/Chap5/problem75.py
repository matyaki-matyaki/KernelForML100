"""問題75"""
def HSIC_2(x, y, z, k_x, k_y, k_z):
    n = len(x)
    S = 0
    for i in range(n):
        for j in range(n):
            S = S + k_x(x[i], x[j]) * k_y(y[i], y[j]) * k_z(z[i], z[j])
    T = 0
    for i in range(n):
        T_1 = 0
        for j in range(n):
            T_1 = T_1 + k_x(x[i], x[j])
        T_2 = 0
        for l in range(n):
            T_2 = T_2 + k_y(y[i], y[l]) * k_z(z[i], z[j])
        T = T + T_1 * T_2
    U = 0
    for i in range(n):
        for j in range(n):
            U = U + k_x(x[i], x[j])
    V = 0
    for i in range(n):
        for j in range(n):
            V = V + k_y(y[i], y[j]) * k_z(z[i], z[j])
    return S / n**2 - 2 * T / n**3 + U * V / n**4

if __name__ == "__main__":
    import numpy as np

    def k_x(x, y):
        return np. exp(-np.linalg.norm(x - y)**2 / 2)

    k_y = k_x
    k_z = k_x

    n = 100
    rs = np.random.RandomState(42)
    x = rs.randn(n)
    y = 2 * x + rs.randn(n)
    z = -3 * x + rs.randn(n)

    print(HSIC_2(x, y, z, k_x, k_y, k_z))
    # 0.0007463123042284747
