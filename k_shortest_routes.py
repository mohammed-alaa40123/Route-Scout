
#!/usr/bin/env python3
"""
k_shortest_routes.py

Usage example:
  python k_shortest_routes.py \
      --src 0 --target 5 --k 5 \
      --graph_attr_path graph_attr.txt \
      --metric delay \
      --traffic_mode all_multiplexed
"""

import argparse
import heapq
from collections import defaultdict
import re
import numpy as np
from typing import Dict, List, Tuple

# ============================================================
#  PARSE graph_attr.txt
# ============================================================
def parse_graph_attr(path: str) -> Tuple[List[int], List[Tuple[int, int, float]]]:
    """
    Parses the GML-like graph_attr.txt and extracts:
      - list of node ids
      - list of edges (src, dst, bandwidth)
    """
    with open(path, "r") as f:
        text = f.read()

    nodes = set()
    edges = []

    # Parse node blocks
    for m in re.finditer(r'node\s*\[\s*(.*?)\s*\]', text, flags=re.S):
        block = m.group(1)
        m_id = re.search(r'\bid\s+([0-9]+)\b', block)
        if m_id:
            nid = int(m_id.group(1))
            nodes.add(nid)

    # Parse edge blocks
    for m in re.finditer(r'edge\s*\[\s*(.*?)\s*\]', text, flags=re.S):
        block = m.group(1)
        m_s = re.search(r'\bsource\s+([0-9]+)\b', block)
        m_t = re.search(r'\btarget\s+([0-9]+)\b', block)
        m_bw = re.search(r'\bbandwidth\s+"?([0-9.]+)"?', block)

        if m_s and m_t and m_bw:
            s = int(m_s.group(1))
            t = int(m_t.group(1))
            bw = float(m_bw.group(1))
            nodes.add(s)
            nodes.add(t)
            edges.append((s, t, bw))

    return sorted(nodes), edges

# ============================================================
#  BUILD GRAPH (cost = 1/bandwidth)
# ============================================================
def build_graph(nodes: List[int], edges: List[Tuple[int, int, float]]):
    """
    Returns adjacency list and adjacency matrix.
    cost = 1.0 / bandwidth
    If bandwidth == 0, the edge is ignored.
    """
    index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # adjacency matrix
    A = np.zeros((N, N), dtype=float)

    # adjacency list: node -> list of (neighbor, cost, bw)
    graph = {n: [] for n in nodes}

    for s, t, bw in edges:
        if bw <= 0:
            # treat bandwidth=0 as no edge; skip
            continue

        cost = 1.0 / bw
        graph[s].append((t, cost, bw))

        i, j = index[s], index[t]
        A[i, j] = cost

    return graph, A, index

# ============================================================
#  SIMPLE k-SHORTEST USING DIJKSTRA + SIDETRACKS (cycle-free)
# ============================================================
def eppstein_k_shortest(graph: Dict[int, List[Tuple[int,float,float]]],
                        source: int, target: int, k: int):
    """
    Simplified k-shortest path enumerator (cycle-free).
    Returns list of (total_cost, path_list).
    """

    # -------------------------
    # Dijkstra for baseline costs
    # -------------------------
    def dijkstra(src):
        dist = {v: float("inf") for v in graph}
        parent = {}
        dist[src] = 0.0
        heap = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, cost, _bw in graph[u]:
                nd = d + cost
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, parent

    dist, parent = dijkstra(source)

    # -------------------------
    # Build sidetracks
    # -------------------------
    sidetracks = defaultdict(list)
    for u in graph:
        for v, cost, _bw in graph[u]:
            if parent.get(v, None) != u:
                du = dist.get(u, float("inf"))
                dv = dist.get(v, float("inf"))
                sidetrack_cost = cost + (dv - du if dv < float("inf") and du < float("inf") else 0.0)
                sidetracks[u].append((v, sidetrack_cost, cost))

    # -------------------------
    # Priority queue search
    # -------------------------
    heap = []
    heapq.heappush(heap, (0.0, source, [source], 0.0))

    results = []
    seen = set()

    while heap and len(results) < k:
        _, u, path, path_cost = heapq.heappop(heap)

        if u == target:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                results.append((path_cost, list(path)))
            continue

        # Expand tree edges
        for v, edge_cost, _bw in graph[u]:
            if parent.get(v, None) == u:
                if v not in path:
                    new_path = path + [v]
                    new_cost = path_cost + edge_cost
                    heapq.heappush(heap, (new_cost, v, new_path, new_cost))

        # Expand sidetracks
        for v, st_cost, edge_cost in sidetracks[u]:
            if v not in path:
                new_path = path + [v]
                new_cost = path_cost + edge_cost
                heapq.heappush(heap, (new_cost, v, new_path, new_cost))

    return results

# ============================================================
#  OUTPUT HELPERS
# ============================================================
def print_and_save_paths(results, src, dst,out_path):
    # out_path = f"candidate_routes_{src}_{dst}.txt"
    with open(out_path, "w") as f:
        for i, (cost, path) in enumerate(results, start=1):
            hops = len(path) - 1
            line = f"{i} (hops={hops}): " + "->".join(str(x) for x in path)
            print(line)
            f.write(line + "\n")
    print(f"\nSaved candidate routes to: {out_path}")

def save_adj_matrix(A, index):
    out_path = "adj_matrix.txt"
    order = [n for n in sorted(index, key=lambda x: index[x])]

    with open(out_path, "w") as f:
        f.write("Adjacency matrix (cost = 1/bandwidth):\n")
        f.write("Node order: " + ", ".join(str(n) for n in order) + "\n\n")
        for i, row in enumerate(A):
            print(f"{i:3d} " + " ".join(f"{v:8.8f}" if v > 0 else "   0.0   " for v in row))

    print(f"Adjacency matrix saved to: {out_path}")

# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="k-shortest paths (cost = 1/bandwidth)")
    parser.add_argument("--src", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--graph_attr_path", type=str, required=True)
    parser.add_argument("--output_path", type=str)
    # parser.add_argument("--metric", type=str, default="delay")
    # parser.add_argument("--traffic_mode", type=str, default="all_multiplexed")
    args = parser.parse_args()

    print("Parsing graph...")
    nodes, edges = parse_graph_attr(args.graph_attr_path)

    graph, A, index = build_graph(nodes, edges)

    # Print + save adjacency matrix
    print("\nAdjacency matrix (cost=1/bandwidth):")
    # print(A)
    save_adj_matrix(A, index)

    print("\nRunning k-shortest path search...")
    results = eppstein_k_shortest(graph, args.src, args.target, args.k)

    if not results:
        print("No path found.")
        return

    print_and_save_paths(results, args.src, args.target,args.output_path)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
