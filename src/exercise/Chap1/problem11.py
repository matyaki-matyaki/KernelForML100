"""問題11"""
import numpy as np


def k(s: list, t: list) -> int:
    """木カーネル。

    木sのi番目の要素s[i]は
    s[i] = [{頂点iのラベル}, [{頂点iの子}]]である。
    """
    m, n = len(s), len(t)
    def C(i: int, j: int) -> int:
        """頂点i, jを根とする共通の部分木の個数。"""
        S, T = s[i], t[j]
        # 木sの頂点iまたは木tの頂点jが子孫を持たない場合0。
        if S[1] is None or T[1] is None:
            return 0
        # 木sの頂点iと木tの頂点jのラベルが一致しないときは0。(1)(a)
        if S[0] != T[0]:
            return 0
        # 子の数が異なるときは0。(1)(b)
        if len(S[1]) != len(T[1]):
            return 0
        U = [s[x][0] for x in S[1]]
        U1 = sorted(U)
        V = [t[y][0] for y in T[1]]
        V1 = sorted(V)
        m = len(U)
        # 子のラベルが一致しないときは0。(1)(c)
        for h in range(m):
            if U1[h] != V1[h]:
                return 0
        U2 = np.array(S[1])[np.argsort(U)]
        V2 = np.array(T[1])[np.argsort(V)]
        W = 1
        for h in range(m):
            W *= 1 + C(U2[h], V2[h])
        return W

    kernel = 0
    for i in range(m):
        for j in range(n):
            if C(i, j) > 0:
                kernel += C(i, j)
    return kernel

if __name__ == "__main__":
    s = [
        ["G", [1, 3]],
        ["T", [2]],
        ["C", None],
        ["A", [4, 5]],
        ["C", None],
        ["T", None],
    ]
    t = [
        ["G", [1, 4]],
        ["A", [2, 3]],
        ["C", None],
        ["T", None],
        ["T", [5, 6]],
        ["C", None],
        ["A", [7, 8]],
        ["C", None],
        ["T", None],
    ]
    print(f"{k(s, s)=}")
    # k(s, s)=6
