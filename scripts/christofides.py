<<<<<<< HEAD
from __future__ import annotations

import time
import tracemalloc
import numpy as np

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

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


# Constrói a MST usando Prim, sem NetworkX.
# Retorna as arestas da MST e os graus dos vértices.
def _compute_mst(
    dist_matrix: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray]:

    n = len(dist_matrix)

    visited = np.zeros(n, dtype=bool)
    min_cost = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)

    min_cost[0] = 0

    mst_edges = []
    degrees = np.zeros(n, dtype=int)

    for _ in range(n):

        # Escolhe o vértice não visitado com menor custo.
        u = np.argmin(np.where(visited, np.inf, min_cost))

        if min_cost[u] == np.inf:
            raise RuntimeError("Não foi possível construir a MST.")

        visited[u] = True

        # Adiciona a aresta que conecta u à MST.
        if parent[u] != -1:
            v = parent[u]

            mst_edges.append((v, u))

            degrees[v] += 1
            degrees[u] += 1

        # Atualiza os custos dos vértices ainda não visitados.
        candidates = np.where(~visited)[0]

        for v in candidates:
            weight = dist_matrix[u, v]

            if weight < min_cost[v]:
                min_cost[v] = weight
                parent[v] = u

    return mst_edges, degrees


# Identifica os vértices de grau ímpar e calcula o MWPM
# usando programação linear inteira via SciPy/HiGHS.
def _compute_odd_vertices_mwpm(
    degrees: np.ndarray,
    dist_matrix: np.ndarray,
) -> list[tuple[int, int]]:

    odd_nodes = np.flatnonzero(degrees % 2 == 1).tolist()

    num_odd = len(odd_nodes)

    if num_odd == 0:
        return []

    if num_odd % 2 != 0:
        raise RuntimeError("Quantidade ímpar de vértices de grau ímpar.")

    # Cada variável representa uma possível aresta do matching.
    edges = []

    for i in range(num_odd):
        u = odd_nodes[i]

        for j in range(i + 1, num_odd):
            v = odd_nodes[j]
            edges.append((u, v))

    num_edges = len(edges)

    # Função objetivo: minimizar a soma dos pesos selecionados.
    c = np.array(
        [dist_matrix[u, v] for u, v in edges],
        dtype=float,
    )

    # Restrição: cada vértice precisa participar de exatamente uma aresta.
    rows = []
    cols = []
    values = []

    odd_index = {
        node: index
        for index, node in enumerate(odd_nodes)
    }

    for edge_index, (u, v) in enumerate(edges):
        rows.extend([odd_index[u], odd_index[v]])
        cols.extend([edge_index, edge_index])
        values.extend([1.0, 1.0])

    A = csr_matrix(
        (values, (rows, cols)),
        shape=(num_odd, num_edges),
    )

    constraints = LinearConstraint(
        A,
        lb=np.ones(num_odd),
        ub=np.ones(num_odd),
    )

    bounds = Bounds(
        lb=np.zeros(num_edges),
        ub=np.ones(num_edges),
    )

    # Variáveis binárias: uma aresta é selecionada ou não.
    result = milp(
        c=c,
        integrality=np.ones(num_edges),
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success or result.x is None:
        raise RuntimeError(
            f"Falha ao obter emparelhamento perfeito: {result.message}"
        )

    matching = [
        edges[i]
        for i in range(num_edges)
        if result.x[i] > 0.5
    ]

    if len(matching) != num_odd // 2:
        raise RuntimeError(
            "Falha ao obter emparelhamento perfeito "
            "sobre os vértices ímpares."
        )

    return matching


# Une MST e matching, encontra o circuito euleriano via Hierholzer
# e aplica shortcutting, sem criar objetos de grafo NetworkX.
def _build_shortcut_tour(
    mst_edges: list[tuple[int, int]],
    matching: list[tuple[int, int]],
    dist_matrix: np.ndarray,
    start_node: int = 0,
) -> tuple[list[int], int]:

    n = len(dist_matrix)

    # Lista de adjacência usando IDs de arestas.
    adjacency = [[] for _ in range(n)]

    edge_id = 0

    # Adiciona as arestas da MST.
    for u, v in mst_edges:
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
        edge_id += 1

    # Adiciona as arestas do matching.
    for u, v in matching:
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
        edge_id += 1

    total_edges = edge_id

    used_edges = np.zeros(total_edges, dtype=bool)
    current_index = np.zeros(n, dtype=int)

    # Hierholzer: constrói o circuito euleriano.
    stack = [start_node]
    eulerian_path = []

    while stack:

        u = stack[-1]

        # Avança até encontrar uma aresta ainda não utilizada.
        while (
            current_index[u] < len(adjacency[u])
            and used_edges[adjacency[u][current_index[u]][1]]
        ):
            current_index[u] += 1

        if current_index[u] == len(adjacency[u]):
            eulerian_path.append(stack.pop())

        else:
            v, eid = adjacency[u][current_index[u]]
            current_index[u] += 1

            if not used_edges[eid]:
                used_edges[eid] = True
                stack.append(v)

    eulerian_path.reverse()

    # Shortcutting: remove visitas repetidas mantendo a ordem.
    visited = np.zeros(n, dtype=bool)
    tour = []

    for node in eulerian_path:
        if not visited[node]:
            visited[node] = True
            tour.append(node)

    # Calcula o custo do ciclo final.
    tour_cost = sum(
        int(dist_matrix[tour[i], tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )

    return tour, tour_cost


# Executa o núcleo do algoritmo de Christofides.
# Retorna tour, custo e profiling das subrotinas.
def compute_christofides_tour(
    graph: TSPGraph,
) -> tuple[list[int], int, dict]:

    mst, mst_prof = _profile_subroutine(
        _compute_mst,
        graph.dist_matrix,
    )

    mst_edges, degrees = mst

    matching, mwpm_prof = _profile_subroutine(
        _compute_odd_vertices_mwpm,
        degrees,
        graph.dist_matrix,
    )

    (tour, tour_cost), euler_prof = _profile_subroutine(
        _build_shortcut_tour,
        mst_edges,
        matching,
        graph.dist_matrix,
        0,
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


# Executa Christofides, valida timeout e retorna
# tempo, fator de aproximação e profiling.
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
            f"Algoritmo de Christofides excedeu o limite "
            f"de {time_limit}s ({total_time:.2f}s)."
        )

    approx_factor = float(tour_cost) / float(optimal_cost)

    if verbose:
        mwpm = profiling["minimum_weight_perfect_matching"]

        print(
            f"[Christofides   | {graph.name}] "
            f"Tempo: {total_time:.6f}s | "
            f"Custo: {tour_cost} | "
            f"Fator Aprox: {approx_factor:.4f} | "
            f"MWPM: {mwpm['time_sec']:.6f}s "
            f"({mwpm['peak_memory_bytes'] / 1024:.1f} KB)"
        )

=======
from __future__ import annotations

import time
import tracemalloc
import numpy as np

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

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


# Constrói a MST usando Prim, sem NetworkX.
# Retorna as arestas da MST e os graus dos vértices.
def _compute_mst(
    dist_matrix: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray]:

    n = len(dist_matrix)

    visited = np.zeros(n, dtype=bool)
    min_cost = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)

    min_cost[0] = 0

    mst_edges = []
    degrees = np.zeros(n, dtype=int)

    for _ in range(n):

        # Escolhe o vértice não visitado com menor custo.
        u = np.argmin(np.where(visited, np.inf, min_cost))

        if min_cost[u] == np.inf:
            raise RuntimeError("Não foi possível construir a MST.")

        visited[u] = True

        # Adiciona a aresta que conecta u à MST.
        if parent[u] != -1:
            v = parent[u]

            mst_edges.append((v, u))

            degrees[v] += 1
            degrees[u] += 1

        # Atualiza os custos dos vértices ainda não visitados.
        candidates = np.where(~visited)[0]

        for v in candidates:
            weight = dist_matrix[u, v]

            if weight < min_cost[v]:
                min_cost[v] = weight
                parent[v] = u

    return mst_edges, degrees


# Identifica os vértices de grau ímpar e calcula o MWPM
# usando programação linear inteira via SciPy/HiGHS.
def _compute_odd_vertices_mwpm(
    degrees: np.ndarray,
    dist_matrix: np.ndarray,
) -> list[tuple[int, int]]:

    odd_nodes = np.flatnonzero(degrees % 2 == 1).tolist()

    num_odd = len(odd_nodes)

    if num_odd == 0:
        return []

    if num_odd % 2 != 0:
        raise RuntimeError("Quantidade ímpar de vértices de grau ímpar.")

    # Cada variável representa uma possível aresta do matching.
    edges = []

    for i in range(num_odd):
        u = odd_nodes[i]

        for j in range(i + 1, num_odd):
            v = odd_nodes[j]
            edges.append((u, v))

    num_edges = len(edges)

    # Função objetivo: minimizar a soma dos pesos selecionados.
    c = np.array(
        [dist_matrix[u, v] for u, v in edges],
        dtype=float,
    )

    # Restrição: cada vértice precisa participar de exatamente uma aresta.
    rows = []
    cols = []
    values = []

    odd_index = {
        node: index
        for index, node in enumerate(odd_nodes)
    }

    for edge_index, (u, v) in enumerate(edges):
        rows.extend([odd_index[u], odd_index[v]])
        cols.extend([edge_index, edge_index])
        values.extend([1.0, 1.0])

    A = csr_matrix(
        (values, (rows, cols)),
        shape=(num_odd, num_edges),
    )

    constraints = LinearConstraint(
        A,
        lb=np.ones(num_odd),
        ub=np.ones(num_odd),
    )

    bounds = Bounds(
        lb=np.zeros(num_edges),
        ub=np.ones(num_edges),
    )

    # Variáveis binárias: uma aresta é selecionada ou não.
    result = milp(
        c=c,
        integrality=np.ones(num_edges),
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success or result.x is None:
        raise RuntimeError(
            f"Falha ao obter emparelhamento perfeito: {result.message}"
        )

    matching = [
        edges[i]
        for i in range(num_edges)
        if result.x[i] > 0.5
    ]

    if len(matching) != num_odd // 2:
        raise RuntimeError(
            "Falha ao obter emparelhamento perfeito "
            "sobre os vértices ímpares."
        )

    return matching


# Une MST e matching, encontra o circuito euleriano via Hierholzer
# e aplica shortcutting, sem criar objetos de grafo NetworkX.
def _build_shortcut_tour(
    mst_edges: list[tuple[int, int]],
    matching: list[tuple[int, int]],
    dist_matrix: np.ndarray,
    start_node: int = 0,
) -> tuple[list[int], int]:

    n = len(dist_matrix)

    # Lista de adjacência usando IDs de arestas.
    adjacency = [[] for _ in range(n)]

    edge_id = 0

    # Adiciona as arestas da MST.
    for u, v in mst_edges:
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
        edge_id += 1

    # Adiciona as arestas do matching.
    for u, v in matching:
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
        edge_id += 1

    total_edges = edge_id

    used_edges = np.zeros(total_edges, dtype=bool)
    current_index = np.zeros(n, dtype=int)

    # Hierholzer: constrói o circuito euleriano.
    stack = [start_node]
    eulerian_path = []

    while stack:

        u = stack[-1]

        # Avança até encontrar uma aresta ainda não utilizada.
        while (
            current_index[u] < len(adjacency[u])
            and used_edges[adjacency[u][current_index[u]][1]]
        ):
            current_index[u] += 1

        if current_index[u] == len(adjacency[u]):
            eulerian_path.append(stack.pop())

        else:
            v, eid = adjacency[u][current_index[u]]
            current_index[u] += 1

            if not used_edges[eid]:
                used_edges[eid] = True
                stack.append(v)

    eulerian_path.reverse()

    # Shortcutting: remove visitas repetidas mantendo a ordem.
    visited = np.zeros(n, dtype=bool)
    tour = []

    for node in eulerian_path:
        if not visited[node]:
            visited[node] = True
            tour.append(node)

    # Calcula o custo do ciclo final.
    tour_cost = sum(
        int(dist_matrix[tour[i], tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )

    return tour, tour_cost


# Executa o núcleo do algoritmo de Christofides.
# Retorna tour, custo e profiling das subrotinas.
def compute_christofides_tour(
    graph: TSPGraph,
) -> tuple[list[int], int, dict]:

    mst, mst_prof = _profile_subroutine(
        _compute_mst,
        graph.dist_matrix,
    )

    mst_edges, degrees = mst

    matching, mwpm_prof = _profile_subroutine(
        _compute_odd_vertices_mwpm,
        degrees,
        graph.dist_matrix,
    )

    (tour, tour_cost), euler_prof = _profile_subroutine(
        _build_shortcut_tour,
        mst_edges,
        matching,
        graph.dist_matrix,
        0,
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


# Executa Christofides, valida timeout e retorna
# tempo, fator de aproximação e profiling.
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
            f"Algoritmo de Christofides excedeu o limite "
            f"de {time_limit}s ({total_time:.2f}s)."
        )

    approx_factor = float(tour_cost) / float(optimal_cost)

    if verbose:
        mwpm = profiling["minimum_weight_perfect_matching"]

        print(
            f"[Christofides   | {graph.name}] "
            f"Tempo: {total_time:.6f}s | "
            f"Custo: {tour_cost} | "
            f"Fator Aprox: {approx_factor:.4f} | "
            f"MWPM: {mwpm['time_sec']:.6f}s "
            f"({mwpm['peak_memory_bytes'] / 1024:.1f} KB)"
        )

>>>>>>> 3966a746e9b81a733c8665db68007f56f7c3d3b8
    return total_time, approx_factor, profiling