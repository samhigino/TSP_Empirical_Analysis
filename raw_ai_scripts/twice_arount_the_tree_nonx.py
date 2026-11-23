from __future__ import annotations

import time
import tracemalloc
import numpy as np

from data_process import TSPGraph


# Executa uma subrotina isoladamente medindo tempo de parede e pico de memória alocada.
def _profile_subroutine(func, *args, **kwargs):
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()

    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()

    if not was_tracing:
        tracemalloc.stop()

    profile = {
        "time_sec": elapsed,
        "peak_memory_bytes": int(peak_bytes),
    }
    return result, profile


# Constrói a Árvore Geradora Mínima (MST) via algoritmo de Prim sobre a matriz de distâncias.
def _compute_mst(dist_matrix: np.ndarray) -> list[list[int]]:
    n = dist_matrix.shape[0]
    adj: list[list[int]] = [[] for _ in range(n)]
    if n <= 1:
        return adj

    in_mst = [False] * n
    min_weight = [float("inf")] * n
    parent = [-1] * n
    min_weight[0] = 0

    # Implementação de Prim baseada em vetores: complexidade O(N^2), ideal para grafos completos.
    for _ in range(n):
        u = -1
        best_w = float("inf")
        for v in range(n):
            if not in_mst[v] and min_weight[v] < best_w:
                best_w = min_weight[v]
                u = v

        in_mst[u] = True
        if parent[u] != -1:
            adj[parent[u]].append(u)
            adj[u].append(parent[u])

        row_u = dist_matrix[u]
        for v in range(n):
            if not in_mst[v]:
                w = int(row_u[v])
                if w < min_weight[v]:
                    min_weight[v] = w
                    parent[v] = u

    return adj


# Realiza o percurso DFS em pré-ordem na MST (shortcutting) e calcula o custo do ciclo hamiltoniano.
def _preorder_tour_and_cost(
    mst_adj: list[list[int]], dist_matrix: np.ndarray, start_node: int = 0
) -> tuple[list[int], int]:
    n = len(mst_adj)
    if n == 0:
        return [], 0

    visited = [False] * n
    tour: list[int] = []
    stack = [start_node]

    # Travessia DFS iterativa em pré-ordem; empilha vizinhos em ordem reversa para manter a ordem natural.
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        tour.append(u)

        for v in reversed(mst_adj[u]):
            if not visited[v]:
                stack.append(v)

    total_cost = sum(int(dist_matrix[tour[i], tour[(i + 1) % n]]) for i in range(n))
    return tour, total_cost


# Executa o núcleo do Twice-Around-the-Tree retornando o tour, seu custo e o profiling das subrotinas.
def compute_twice_around_tour(graph: TSPGraph) -> tuple[list[int], int, dict]:
    mst_adj, mst_prof = _profile_subroutine(_compute_mst, graph.dist_matrix)
    (tour, tour_cost), dfs_prof = _profile_subroutine(
        _preorder_tour_and_cost, mst_adj, graph.dist_matrix, 0
    )

    profiling = {
        "mst_construction": mst_prof,
        "preorder_shortcutting": dfs_prof,
        "total_peak_memory_bytes": max(
            mst_prof["peak_memory_bytes"], dfs_prof["peak_memory_bytes"]
        ),
    }
    return tour, tour_cost, profiling


# Executa o Twice-Around-the-Tree, verifica o timeout de 30 min e retorna tempo, fator de aproximação e profiling.
def twice_around_the_tree(
    graph: TSPGraph,
    optimal_cost: float,
    time_limit: float = 1800.0,
    verbose: bool = True,
) -> tuple[float, float, dict]:
    start_time = time.perf_counter()
    _, tour_cost, profiling = compute_twice_around_tour(graph)
    total_time = time.perf_counter() - start_time

    if total_time > time_limit:
        raise TimeoutError(
            f"Twice-Around-the-Tree excedeu o limite de {time_limit}s ({total_time:.2f}s)."
        )

    approx_factor = float(tour_cost) / float(optimal_cost)

    if verbose:
        print(
            f"[Twice-Around | {graph.name}] Tempo: {total_time:.6f}s | "
            f"Custo: {tour_cost} | Fator Aprox: {approx_factor:.4f} | "
            f"MST: {profiling['mst_construction']['time_sec']:.6f}s "
            f"({profiling['mst_construction']['peak_memory_bytes'] / 1024:.1f} KB)"
        )

    return total_time, approx_factor, profiling