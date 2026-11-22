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


# Constrói a Árvore Geradora Mínima (MST) do grafo completo usando o algoritmo de Prim.
def _compute_mst(nx_graph: nx.Graph) -> nx.Graph:
    return nx.minimum_spanning_tree(nx_graph, weight="weight", algorithm="prim")

# Realiza o percurso DFS em pré-ordem na MST (shortcutting) e calcula o custo do ciclo hamiltoniano.
def _preorder_tour_and_cost(
    mst: nx.Graph, dist_matrix: np.ndarray, start_node: int = 0
) -> tuple[list[int], int]:
    # A travessia em pré-ordem da árvore corresponde a duplicar arestas e saltar vértices já visitados.
    tour = list(nx.dfs_preorder_nodes(mst, source=start_node))
    n = len(tour)
    total_cost = sum(int(dist_matrix[tour[i], tour[(i + 1) % n]]) for i in range(n))
    return tour, total_cost


# Executa o núcleo do Twice-Around-the-Tree retornando o tour, seu custo e o profiling das subrotinas.
def compute_twice_around_tour(graph: TSPGraph) -> tuple[list[int], int, dict]:
    mst, mst_prof = _profile_subroutine(_compute_mst, graph.nx_graph)
    (tour, tour_cost), dfs_prof = _profile_subroutine(
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


# Executa o Twice-Around-the-Tree, verifica o timeout de 30 min e retorna tempo, fator de aproximação e profiling.
def twice_around_the_tree(
    graph: TSPGraph,                # Grafo do problema no formato TSPGraph (só usaremos a matriz)
    optimal_cost: float,            # Custo do percurso ótimo previamente conhecido
    time_limit: float = 1800.0,     # Tempo limite (30 minutos)
    debug: bool = True,             # Caso queiramos imprimir os resultados no terminal
) -> tuple[float, float, dict]:
    
    start_time = time.perf_counter()
    _, tour_cost, profiling = compute_twice_around_tour(graph)
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