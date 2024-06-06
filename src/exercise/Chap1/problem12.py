"""問題12"""

def string_kernel(x: str, y: str) -> float:
    """String kernel."""
    m, n = len(x), len(y)
    S = 0
    for i in range(m):
        for j in range(i, m):
            for k in range(n):
                if x[i:j+1] == y[k:k + j - i + 1]:
                    S += 1
    return S

if __name__ == "__main__":
    import numpy as np
    rs = np.random.RandomState(42)
    x = "".join(rs.choice(["0", "1"], size=10))
    y = "".join(rs.choice(["0", "1"], size=10))
    print(f"{x=}")
    # x='0100010001'
    print(f"{y=}")
    # y='0000101110'
    print(f"{string_kernel(x, y)=}")
    # string_kernel(x, y)=88
