from __future__ import annotations

import time
import tracemalloc
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from christofides import compute_christofides_tour
from data_process import TSPGraph
from twice_around_the_tree import compute_twice_around_tour


# Calcula o custo total de um ciclo hamiltoniano fechado a partir da matriz de distâncias.
def _compute_tour_cost(tour: list[int], dist_matrix: np.ndarray) -> int:
    n = len(tour)
    return sum(int(dist_matrix[tour[i], tour[(i + 1) % n]]) for i in range(n))


# Heurística 1 (Trivial): gera o ciclo canônico 0 -> 1 -> ... -> N-1 -> 0 e retorna tour e custo.
def heuristic_trivial(graph: TSPGraph) -> tuple[list[int], int]:
    tour = list(range(graph.dimension))
    return tour, _compute_tour_cost(tour, graph.dist_matrix)


# Heurística 2 (Busca Gulosa): constrói um ciclo via Vizinho Mais Próximo a partir do vértice 0.
def heuristic_nearest_neighbor(graph: TSPGraph) -> tuple[list[int], int]:
    n = graph.dimension
    dist = graph.dist_matrix
    visited = [False] * n
    tour = [0]
    visited[0] = True
    curr = 0

    for _ in range(n - 1):
        best_next = -1
        best_dist = float("inf")
        for candidate in range(n):
            if not visited[candidate] and dist[curr, candidate] < best_dist:
                best_dist = dist[curr, candidate]
                best_next = candidate
        visited[best_next] = True
        tour.append(best_next)
        curr = best_next

    return tour, _compute_tour_cost(tour, dist)


# Obtém coordenadas 2D da instância ou projeta a matriz de distâncias via MDS clássico para o fecho convexo.
def _get_or_embed_2d_coords(graph: TSPGraph) -> np.ndarray:
    if graph.coords is not None:
        return graph.coords

    # Para instâncias MATRIX puras (ex.: gr17), aplica Classical MDS (Torgerson-Gower) em R^2.
    d2 = graph.dist_matrix.astype(np.float64) ** 2
    n = graph.dimension
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * (h @ d2 @ h)
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1][:2]
    vals = np.maximum(eigvals[idx], 0.0)
    return eigvecs[:, idx] * np.sqrt(vals)


# Heurística 3 (Convex Hull): inicializa o ciclo com o fecho convexo e insere vértices internos gulosamente.
def heuristic_convex_hull(graph: TSPGraph) -> tuple[list[int], int]:
    n = graph.dimension
    dist = graph.dist_matrix
    if n <= 3:
        tour = list(range(n))
        return tour, _compute_tour_cost(tour, dist)

    coords_2d = _get_or_embed_2d_coords(graph)
    try:
        hull = ConvexHull(coords_2d)
        subtour = [int(v) for v in hull.vertices]
    except QhullError:
        # Fallback seguro caso os pontos projetados sejam colineares.
        far_node = int(np.argmax(dist[0]))
        subtour = [0, far_node]

    in_tour = set(subtour)
    unvisited = [v for v in range(n) if v not in in_tour]

    # Inserção mais barata (Cheapest Insertion): escolhe o par (vértice interno, aresta) de menor acréscimo.
    while unvisited:
        best_node = -1
        best_pos = -1
        best_delta = float("inf")
        m = len(subtour)

        for r in unvisited:
            for idx in range(m):
                i = subtour[idx]
                j = subtour[(idx + 1) % m]
                delta = int(dist[i, r]) + int(dist[r, j]) - int(dist[i, j])
                if delta < best_delta:
                    best_delta = delta
                    best_node = r
                    best_pos = idx + 1

        subtour.insert(best_pos, best_node)
        unvisited.remove(best_node)

    return subtour, _compute_tour_cost(subtour, dist)


# Heurística 4 (Twice-Around-the-Tree): usa o custo da aproximação Twice-Around-the-Tree como limite inicial.
def heuristic_twice_around_the_tree(graph: TSPGraph) -> tuple[list[int], int]:
    tour, cost, _ = compute_twice_around_tour(graph)
    return tour, cost


# Heurística 5 (Christofides): usa o custo da aproximação de Christofides como limite inicial.
def heuristic_christofides(graph: TSPGraph) -> tuple[list[int], int]:
    tour, cost, _ = compute_christofides_tour(graph)
    return tour, cost


# Seleciona e executa a estratégia de warm-start especificada, medindo seu tempo de execução.
def compute_initial_upper_bound(graph: TSPGraph, strategy: str) -> tuple[list[int], int, float]:
    strategies = {
        "trivial": heuristic_trivial,
        "nearest_neighbor": heuristic_nearest_neighbor,
        "convex_hull": heuristic_convex_hull,
        "twice_around_the_tree": heuristic_twice_around_the_tree,
        "christofides": heuristic_christofides,
    }
    key = strategy.strip().lower()
    if key not in strategies:
        raise ValueError(
            f"Estratégia de warm-start inválida: '{strategy}'. "
            f"Opções válidas: {list(strategies.keys())}"
        )

    t0 = time.perf_counter()
    tour, upper_bound = strategies[key](graph)
    heuristic_time = time.perf_counter() - t0
    return tour, upper_bound, heuristic_time


# Calcula o limite inferior (Lower Bound) admissível baseado nas duas menores arestas incidentes por vértice.
def _compute_lower_bound(
    path_cost: int,
    curr: int,
    visited_mask: int,
    dist: list[list[int]],
    sorted_nbrs: list[tuple[int, ...]],
) -> int:
    n = len(dist)
    # Vértices internos do caminho parcial já possuem grau 2 fixado (suas 2 arestas já estão em 2*path_cost).
    internal_mask = visited_mask & ~(1 | (1 << curr))
    total_double_bound = 2 * path_cost

    # Arestas remanescentes mínimas conectando as extremidades (0 e curr) a vértices ainda não visitados.
    for u in sorted_nbrs[0]:
        if not (visited_mask & (1 << u)):
            total_double_bound += dist[0][u]
            break

    for u in sorted_nbrs[curr]:
        if not (visited_mask & (1 << u)):
            total_double_bound += dist[curr][u]
            break

    # Para cada vértice não visitado, soma as duas menores arestas para vértices não-internos.
    for w in range(1, n):
        if not (visited_mask & (1 << w)):
            found = 0
            for u in sorted_nbrs[w]:
                if not (internal_mask & (1 << u)):
                    total_double_bound += dist[w][u]
                    found += 1
                    if found == 2:
                        break

    # Como todos os pesos são inteiros, teto(soma / 2) é um limite inferior admissível exato.
    return (total_double_bound + 1) // 2


# Executa o algoritmo exato Branch-and-Bound (DFS guiada pelo menor limitante) com limite de 30 minutos.
def branch_and_bound(
    graph: TSPGraph,
    warm_start: str = "christofides",
    time_limit: float = 1800.0,
) -> dict:
    tracemalloc.start()
    tracemalloc.reset_peak()
    overall_start = time.perf_counter()

    # 1. Geração do limite superior inicial (Upper Bound) via heurística escolhida.
    best_tour, best_cost, heuristic_time = compute_initial_upper_bound(graph, warm_start)

    n = graph.dimension
    if n <= 3:
        total_time = time.perf_counter() - overall_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "warm_start": warm_start,
            "total_time_sec": total_time,
            "heuristic_time_sec": heuristic_time,
            "best_cost": best_cost,
            "best_tour": best_tour,
            "is_optimal": True,
            "timed_out": False,
            "nodes_expanded": 0,
            "nodes_pruned": 0,
            "peak_memory_bytes": int(peak_mem),
        }

    # Estruturas nativas em lista/tupla para máxima velocidade no laço interno do Python 3.13.
    dist: list[list[int]] = graph.dist_matrix.tolist()
    sorted_nbrs: list[tuple[int, ...]] = [
        tuple(sorted((u for u in range(n) if u != v), key=lambda u, v=v: dist[v][u]))
        for v in range(n)
    ]

    nodes_expanded = 0
    nodes_pruned = 0
    timed_out = False

    # 2. Inicialização da pilha de Busca em Profundidade (DFS) fixando a origem no vértice 0.
    # Cada estado na pilha é: (curr_city, depth, path_cost, visited_mask, first_city).
    # Quebra de simetria especular: em um TSP simétrico, basta exigir first_city < last_city.
    stack: list[tuple[int, int, int, int, int]] = []
    initial_children: list[tuple[int, int, int, int, int, int]] = []

    for v1 in range(1, n):
        # Como v1 < last_city é exigido ao fechar o ciclo, v1 = n - 1 nunca pode ser o primeiro passo.
        if v1 == n - 1:
            continue
        mask = 1 | (1 << v1)
        c1 = dist[0][v1]
        lb = _compute_lower_bound(c1, v1, mask, dist, sorted_nbrs)
        if lb >= best_cost:
            nodes_pruned += 1
        else:
            initial_children.append((lb, v1, 1, c1, mask, v1))

    # Empilha em ordem decrescente de LB para que o filho com MENOR LB seja desempilhado primeiro (DFS guiada).
    initial_children.sort(key=lambda item: item[0], reverse=True)
    for _, curr, depth, p_cost, mask, first_c in initial_children:
        stack.append((curr, depth, p_cost, mask, first_c))

    full_mask = (1 << n) - 1
    check_interval = 1024
    iter_count = 0

    while stack:
        iter_count += 1
        if iter_count % check_interval == 0:
            if (time.perf_counter() - overall_start) >= time_limit:
                timed_out = True
                break

        curr, depth, path_cost, visited_mask, first_city = stack.pop()
        nodes_expanded += 1

        # Caso base (penúltimo nível): apenas 1 vértice permanece não visitado; avalia fechamento em O(1).
        if depth == n - 2:
            rem_bit = full_mask ^ visited_mask
            last_city = rem_bit.bit_length() - 1
            if first_city < last_city:
                total_tour_cost = path_cost + dist[curr][last_city] + dist[last_city][0]
                if total_tour_cost < best_cost:
                    best_cost = total_tour_cost
                else:
                    nodes_pruned += 1
            else:
                nodes_pruned += 1
            continue

        # Expansão dos filhos válidos ordenados pelo menor limite inferior (Dive-First / Best-Child DFS).
        children: list[tuple[int, int, int, int, int, int]] = []
        for nxt in sorted_nbrs[curr]:
            if visited_mask & (1 << nxt):
                continue
            new_cost = path_cost + dist[curr][nxt]
            if new_cost >= best_cost:
                nodes_pruned += 1
                continue

            new_mask = visited_mask | (1 << nxt)
            lb = _compute_lower_bound(new_cost, nxt, new_mask, dist, sorted_nbrs)
            if lb >= best_cost:
                nodes_pruned += 1
            else:
                children.append((lb, nxt, depth + 1, new_cost, new_mask, first_city))

        children.sort(key=lambda item: item[0], reverse=True)
        for _, nxt, new_depth, new_cost, new_mask, f_city in children:
            stack.append((nxt, new_depth, new_cost, new_mask, f_city))

    total_time = time.perf_counter() - overall_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Tratamento explícito do caso especial de estouro do limite de tempo (30 minutos).
    if timed_out or total_time >= time_limit:
        timed_out = True
        total_time = max(total_time, time_limit)

    return {
        "warm_start": warm_start,
        "total_time_sec": total_time,
        "heuristic_time_sec": heuristic_time,
        "best_cost": int(best_cost),
        "is_optimal": not timed_out,
        "timed_out": timed_out,
        "nodes_expanded": nodes_expanded,
        "nodes_pruned": nodes_pruned,
        "peak_memory_bytes": int(peak_mem),
    }