from __future__ import annotations

import time
import tracemalloc
import numpy as np

from data_process import TSPGraph

# -------------------------------------------------------------
# Este módulo implementa o twice-around-the-tree sem usar
# a biblioteca networkx
# -------------------------------------------------------------

# Executa uma subrotina isoladamente medindo tempo de parede e pico de memória alocada.
def profile_subroutine(func, *args, **kwargs):
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


# Constrói a Árvore Geradora Mínima (MST) do grafo completo usando o algoritmo de Prim (a MST é retornada em lista de adjacência)
def compute_mst(dist_matrix: np.ndarray) -> list[list[int]]:
    n = dist_matrix.shape[0]
    adj: list[list[int]] = [[] for _ in range(n)]
    if n <= 1:
        raise RuntimeError("Instância vazia ou com 1 cidade detectada em twice_around_the_tree.")

    in_mst = [False] * n
    min_weight = [float("inf")] * n
    parent = [-1] * n
    min_weight[0] = 0

    # Algoritmo de Prim
    for i in range(n):
        u = -1
        best_w = float("inf")
        # Escolhe a melhor aresta que conecta um vértice novo para entrar na solução
        for v in range(n):
            if not in_mst[v] and min_weight[v] < best_w:
                best_w = min_weight[v]
                u = v

        # Insere a aresta na solução
        in_mst[u] = True
        if parent[u] != -1:
            adj[parent[u]].append(u)
            adj[u].append(parent[u])

        # Atualiza pesos mínimos e pais com base na aresta nova
        row_u = dist_matrix[u]
        for v in range(n):
            if not in_mst[v]:
                w = int(row_u[v])
                if w < min_weight[v]:
                    min_weight[v] = w
                    parent[v] = u

    return adj

# Realiza o percurso DFS em pré-ordem na MST (shortcutting) e calcula o custo do ciclo hamiltoniano.
def _preorder_tour_and_cost(mst: list[list[int]], dist_matrix: np.ndarray, start_node: int = 0) -> tuple[list[int], int]:
    n = len(mst)
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

        for v in reversed(mst[u]):
            if not visited[v]:
                stack.append(v)

    total_cost = sum(int(dist_matrix[tour[i], tour[(i + 1) % n]]) for i in range(n))
    return tour, total_cost


def compute_twice_around_tour_full(graph: TSPGraph) -> tuple[list[int], int, dict]:
    mst, mst_prof = profile_subroutine(compute_mst, graph.dist_matrix)
    (tour, tour_cost), dfs_prof = profile_subroutine(
        _preorder_tour_and_cost, mst, graph.dist_matrix, 0
    )

    profiling = {
        "mst_construction": mst_prof,
        "preorder_shortcutting": dfs_prof,
        "total_peak_memory_bytes": max(
            mst_prof["peak_memory_bytes"], dfs_prof["peak_memory_bytes"]
        ),
    }

    return tour, tour_cost, profiling


def compute_twice_around_tour(graph: TSPGraph) -> tuple[int, dict]:
    _, tour_cost, profiling = compute_twice_around_tour_full(graph)
    return tour_cost, profiling


# Executa o Twice-Around-the-Tree, verifica o timeout de 30 min e retorna tempo, fator de aproximação e profiling.
def twice_around_the_tree(
    graph: TSPGraph,                # Grafo do problema no formato TSPGraph (só usaremos a matriz)
    optimal_cost: float,            # Custo do percurso ótimo previamente conhecido
    time_limit: float = 1800.0,     # Tempo limite (30 minutos)
    debug: bool = True,             # Caso queiramos imprimir os resultados no terminal
) -> tuple[float, float, dict]:
    
    start_time = time.perf_counter()
    tour_cost, profiling = compute_twice_around_tour(graph)
    total_time = time.perf_counter() - start_time

    if total_time > time_limit:
        raise TimeoutError(f"Twice-Around-the-Tree excedeu o limite de {time_limit}s ({total_time:.2f}s).")

    approx_factor = float(tour_cost) / float(optimal_cost)

    if debug:
        print(
            f"[Twice-Around | {graph.name}] Tempo: {total_time:.6f}s | "
            f"Custo: {tour_cost} | Fator Aprox: {approx_factor:.4f} | "
            f"MST: {profiling['mst_construction']['time_sec']:.6f}s "
            f"({profiling['mst_construction']['peak_memory_bytes'] / 1024:.1f} KB)"
        )

    return total_time, approx_factor, profiling