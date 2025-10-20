import time
import math
import requests
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ----------------- 可调参数 -----------------
START_ADDRESS = "1dice8EMZmqKvrGE4Qc9bUFf9PX3xaYDp"   # 起始地址（换成你的种子地址）
MAX_DEPTH = 2                 # 向外扩散层数（0=只抓起始地址，1=抓其直接邻居，依此类推）
MAX_TX_PER_ADDRESS = 50       # 每个地址最多抓取多少笔交易（Blockchain.info每次最多50，已做分页）
MAX_NODES = 5000              # 为防止图过大设置上限
REQUEST_SLEEP = 1.0           # API 请求间隔(秒)，避免被限流
OUT_IMAGE = "btc_address_graph.png"
# -------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-address-graph/1.0"})

def fetch_txs_blockchain_info(address: str, limit_each=50, max_txs=200):
    """从 blockchain.info/rawaddr/{address} 抓交易，做分页，返回 tx 列表"""
    txs = []
    offset = 0
    while len(txs) < max_txs:
        url = f"https://blockchain.info/rawaddr/{address}"
        params = {"limit": min(limit_each, 50), "offset": offset}
        resp = SESSION.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("txs", [])
        txs.extend(batch)
        if len(batch) < params["limit"]:
            break
        offset += params["limit"]
        time.sleep(REQUEST_SLEEP)
    return txs[:max_txs]

def address_edges_from_txs(txs):
    """
    把一批交易转成 地址->地址 的边列表（(u, v, weight)）。
    使用 haircut 近似：每个输出金额 val 平均分摊到所有输入。
    """
    edges = []
    for tx in txs:
        inputs = []
        for i in tx.get("inputs", []):
            prev = i.get("prev_out") or {}
            a = prev.get("addr")
            if a: inputs.append(a)

        outs = []
        for o in tx.get("out", []):
            a = o.get("addr")
            val_btc = (o.get("value") or 0) / 1e8
            if a and val_btc > 0:
                outs.append((a, val_btc))

        if not inputs or not outs:
            continue

        denom = max(1, len(inputs))
        for in_a in inputs:
            for out_a, val in outs:
                w = val / denom
                edges.append((in_a, out_a, w))
    return edges

def crawl_graph(seed_address):
    """
    从种子地址做 BFS 抓图，返回 (G, stats)
    G: nx.DiGraph，边权为估计金额
    """
    G = nx.DiGraph()
    visited = set()
    q = deque([(seed_address, 0)])

    edge_w = defaultdict(float)  # ((u,v) -> weight)
    node_amount = defaultdict(float)

    while q and len(G) < MAX_NODES:
        addr, depth = q.popleft()
        if addr in visited:
            continue
        visited.add(addr)
        G.add_node(addr)

        # 抓该地址交易
        try:
            txs = fetch_txs_blockchain_info(addr, max_txs=MAX_TX_PER_ADDRESS)
        except Exception as e:
            print(f"[warn] fetch {addr} failed: {e}")
            continue

        # 交易 -> 边
        edges = address_edges_from_txs(txs)

        # 统计边与邻居
        neigh_to_enqueue = set()
        for u, v, w in edges:
            edge_w[(u, v)] += w
            node_amount[u] += w
            node_amount[v] += w
            if depth < MAX_DEPTH:
                # 只向外扩输出地址（也可把输入也加入，视需求）
                neigh_to_enqueue.add(v)

        # 写入到图（为控规模，只加目前已知节点）
        for (u, v), w in edge_w.items():
            if not G.has_node(u):
                G.add_node(u)
            if not G.has_node(v):
                G.add_node(v)
            G.add_edge(u, v, weight=w)

        # 扩展队列
        for nb in neigh_to_enqueue:
            if nb not in visited and len(G) < MAX_NODES:
                q.append((nb, depth + 1))

        time.sleep(REQUEST_SLEEP)

    return G, {"node_amount": node_amount, "edges_weight": edge_w}

def draw_graph(G, node_amount, out_file=OUT_IMAGE):
    # 节点大小：按“流经该地址的估计金额”与度综合缩放
    amt = {n: node_amount.get(n, 0.0) for n in G.nodes()}
    deg = dict(G.degree())
    sizes = []
    for n in G.nodes():
        s = 30 + 6 * math.sqrt(deg.get(n, 0)) + 4 * math.log1p(amt.get(n, 0.0))
        sizes.append(min(300, s))

    # 环状更明显的 Kamada-Kawai 布局
    print("Computing layout (kamada_kawai)...")
    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(10, 10), dpi=180)
    # 画边：数量过大时采样以提速
    edges = list(G.edges(data=True))
    if len(edges) > 8000:
        step = max(1, len(edges) // 8000)
        edges = edges[::step]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, _ in edges],
        alpha=0.15, arrows=False, width=0.5
    )
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#1740FA")  # 深蓝
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Graph saved to: {out_file}")

def main():
    G, stats = crawl_graph(START_ADDRESS)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    draw_graph(G, stats["node_amount"], OUT_IMAGE)

if __name__ == "__main__":
    main()
