import numpy as np


def incomplete_cholesky(A, m):
    A_copy = A.copy()  # Aのコピーを作成
    n = A_copy.shape[0]
    R = np.zeros((n, n))
    P = np.eye(n)
    for i in range(m):
        max_R = -np.inf
        for j in range(i, n):
            RR = A_copy[j, j]
            for h in range(i):
                RR = RR - R[j, h]**2
            if max_R < RR:
                k = j
                max_R = RR
        R[i, i] = np.sqrt(max_R)
        if k != i:
            for j in range(i):
                w = R[i, j]
                R[i, j] = R[k, j]
                R[k, j] = w
            for j in range(n):
                w = A_copy[j, k]
                A_copy[j, k] = A_copy[j, i]
                A_copy[j, i] = w
            for j in range(n):
                w = A_copy[k, j]
                A_copy[k, j] = A_copy[i, j]
                A_copy[i, j] = w
            Q = np.eye(n)
            Q[i, i] = 0
            Q[k, k] = 0
            Q[i, k] = 1
            Q[k, i] = 1
            P = np.dot(P, Q)
        if i < n - 1:
            for j in range(i + 1, n):
                S = A_copy[j, i]
                for h in range(i):
                    S = S - R[i, h] * R[j, h]
                R[j, i] = S / R[i, i]
    return np.dot(P, R)

if __name__ == "__main__":
    n = 5
    r = 3
    D = np.random.randint(-n, n, size=(n, n))
    A = np.dot(D, D.T) + n * np.eye(n)  # A is symmetric positive definite
    L = incomplete_cholesky(A, r)
    print("A\n", A)
    """
    A
    [[ 45.  46.  28. -15.  -5.]
    [ 46.  72.  28.  -5.   2.]
    [ 28.  28.  67.  15.  22.]
    [-15.  -5.  15.  54.  39.]
    [ -5.   2.  22.  39.  64.]]
    """
    print("L @ L.T\n", np.dot(L, L.T))
    """
    L @ L.T
    [[33.03589072 46.         28.         -6.05404855 -5.        ]
    [46.         72.         28.         -5.          2.        ]
    [28.         28.         67.         15.         22.        ]
    [-6.05404855 -5.         15.         24.62197001 39.        ]
    [-5.          2.         22.         39.         64.        ]]
    """
