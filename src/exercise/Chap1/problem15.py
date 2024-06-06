"""問題15"""

def prob(node: list[list[int]], s: list[int], p: float) -> float:
    """Prob."""
    if len(node[s[0]]) == 0:
        return 0
    m = len(s)
    if m == 1:
        return p
    return (1 - p) / len(node[s[0]]) * prob(node, s[1:m], p)

def k(node: list[list[int]], s: list[int], p: float) -> float:
    """Graph kernel."""
    return prob(node, s, p) / len(node)

if __name__ == "__main__":
    node = [
        [1, 3],
        [3],
        [0, 4],
        [2],
        [2],
    ]
    # 1-indexedとして経路1->3は存在し得ないものである。（頂点1から3への有向辺は存在しないので。）
    # このとき、p(\pi = <1, 3>))=0となるべきである。
    print(f"{k(node, [0, 2], 1/2)=}")
    # 0.025
    # しかし上記の実行結果のように、非ゼロの値が得られる。
