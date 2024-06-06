"""問題12（誤植訂正後）"""

def string_kernel(x: str, y: str, p: int) -> int:
    """String kernel."""
    m, n = len(x), len(y)
    if m < p or n < p:
        return 0
    S = 0
    for i in range(m - p):
        for j in range(n - p):
            if x[i:i + p] == y[j:j + p]:
                S += 1
    return S

if __name__ == "__main__":
    import numpy as np
    rs = np.random.RandomState(42)
    x = "".join(rs.choice(["0", "1"], size=10))
    y = "".join(rs.choice(["0", "1"], size=10))
    print(f"{x=}")
    print(f"{y=}")
    print(f"{string_kernel(x, y, 2)=}")
    # string_kernel(x, y, 2)=18
