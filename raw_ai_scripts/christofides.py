from __future__ import annotations

import time
import tracemalloc
import networkx as nx
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


# Constrói a Árvore Geradora Mínima (MST) do grafo completo usando o algoritmo de Prim.
def _compute_mst(nx_graph: nx.Graph) -> nx.Graph:
    return nx.minimum_spanning_tree(nx_graph, weight="weight", algorithm="prim")


# Identifica vértices de grau ímpar na MST e calcula o Emparelhamento Perfeito de Peso Mínimo (Blossom).
def _compute_odd_vertices_mwpm(
    mst: nx.Graph, dist_matrix: np.ndarray
) -> list[tuple[int, int]]:
    odd_nodes = [v for v, deg in mst.degree() if deg % 2 == 1]

    # Subgrafo completo induzido pelos vértices de grau ímpar (sempre em número par).
    odd_subgraph = nx.Graph()
    odd_subgraph.add_nodes_from(odd_nodes)
    num_odd = len(odd_nodes)
    for i in range(num_odd):
        u = odd_nodes[i]
        for j in range(i + 1, num_odd):
            v = odd_nodes[j]
            odd_subgraph.add_edge(u, v, weight=int(dist_matrix[u, v]))

    # Algoritmo Blossom de Edmonds: cardinalidade máxima + peso mínimo garante matching perfeito.
    matching = nx.min_weight_matching(odd_subgraph, weight="weight")
    if len(matching) != num_odd // 2:
        raise RuntimeError("Falha ao obter emparelhamento perfeito sobre os vértices ímpares.")

    return list(matching)


# Une a MST e o matching em um multigrafo euleriano, extrai o circuito euleriano e aplica shortcutting.
def _build_shortcut_tour(
    mst: nx.Graph,
    matching: list[tuple[int, int]],
    dist_matrix: np.ndarray,
    start_node: int = 0,
) -> tuple[list[int], int]:
    eulerian_multigraph = nx.MultiGraph()
    eulerian_multigraph.add_nodes_from(mst.nodes())
    eulerian_multigraph.add_edges_from(mst.edges(data=True))
    for u, v in matching:
        eulerian_multigraph.add_edge(u, v, weight=int(dist_matrix[u, v]))

    # Extrai o circuito euleriano (Hierholzer) e remove revisitas via shortcutting.
    visited = set()
    tour: list[int] = []
    for u, _ in nx.eulerian_circuit(eulerian_multigraph, source=start_node):
        if u not in visited:
            visited.add(u)
            tour.append(u)

    n = len(tour)
    total_cost = sum(int(dist_matrix[tour[i], tour[(i + 1) % n]]) for i in range(n))
    return tour, total_cost


# Executa o núcleo do algoritmo de Christofides retornando o tour, seu custo e o profiling das subrotinas.
def compute_christofides_tour(graph: TSPGraph) -> tuple[list[int], int, dict]:
    mst, mst_prof = _profile_subroutine(_compute_mst, graph.nx_graph)
    matching, mwpm_prof = _profile_subroutine(
        _compute_odd_vertices_mwpm, mst, graph.dist_matrix
    )
    (tour, tour_cost), euler_prof = _profile_subroutine(
        _build_shortcut_tour, mst, matching, graph.dist_matrix, 0
    )

    profiling = {
        "mst_construction": mst_prof,
        "minimum_weight_perfect_matching": mwpm_prof,
        "eulerian_circuit_and_shortcutting": euler_prof,
        "total_peak_memory_bytes": max(
            mst_prof["peak_memory_bytes"],
            mwpm_prof["peak_memory_bytes"],
            euler_prof["peak_memory_bytes"],
        ),
    }
    return tour, tour_cost, profiling


# Executa o algoritmo de Christofides, valida o timeout de 30 min e retorna tempo, fator de aproximação e profiling.
def christofides(
    graph: TSPGraph,
    optimal_cost: float,
    time_limit: float = 1800.0,
    verbose: bool = True,
) -> tuple[float, float, dict]:
    start_time = time.perf_counter()
    _, tour_cost, profiling = compute_christofides_tour(graph)
    total_time = time.perf_counter() - start_time

    if total_time > time_limit:
        raise TimeoutError(
            f"Algoritmo de Christofides excedeu o limite de {time_limit}s ({total_time:.2f}s)."
        )

    approx_factor = float(tour_cost) / float(optimal_cost)

    if verbose:
        mwpm = profiling["minimum_weight_perfect_matching"]
        print(
            f"[Christofides   | {graph.name}] Tempo: {total_time:.6f}s | "
            f"Custo: {tour_cost} | Fator Aprox: {approx_factor:.4f} | "
            f"MWPM: {mwpm['time_sec']:.6f}s ({mwpm['peak_memory_bytes'] / 1024:.1f} KB)"
        )

    return total_time, approx_factor, profiling