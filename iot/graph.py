"""Layer 6 geometry: place 100 virtual sensor nodes over BC and build the
k-nearest-neighbour graph the GAT attends over. Saved to artifacts/graph.json
so the simulator, trainer and stream processor all share one topology.
"""
from __future__ import annotations
import json
import numpy as np
from config import GRID, BC_BOUNDS, GRAPH_KNN, GRAPH_PATH, N_NODES


def node_coords():
    rows, cols = GRID
    lats = np.linspace(BC_BOUNDS["lat_min"], BC_BOUNDS["lat_max"], rows)
    lons = np.linspace(BC_BOUNDS["lon_min"], BC_BOUNDS["lon_max"], cols)
    coords = [(float(la), float(lo)) for la in lats for lo in lons]
    return coords[:N_NODES]


def build_adjacency(coords, k=GRAPH_KNN):
    n = len(coords)
    pts = np.array(coords)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
        nn = np.argsort(d)[1:k + 1]
        adj[i, nn] = 1
        adj[nn, i] = 1            # undirected
    np.fill_diagonal(adj, 1)      # self-loops
    return adj


def save_graph():
    coords = node_coords()
    adj = build_adjacency(coords)
    GRAPH_PATH.write_text(json.dumps({"coords": coords, "adj": adj.tolist()}))
    print(f"[graph] {len(coords)} nodes, k={GRAPH_KNN} -> {GRAPH_PATH}")
    return coords, adj


def load_graph():
    g = json.loads(GRAPH_PATH.read_text())
    return g["coords"], np.array(g["adj"])


if __name__ == "__main__":
    save_graph()
