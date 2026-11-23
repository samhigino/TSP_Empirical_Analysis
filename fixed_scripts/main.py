from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

from branch_and_bound import branch_and_bound
from christofides import christofides
from data_process import load_tsp_instance
from twice_around_the_tree import twice_around_the_tree

# ==============================================================================
# CONFIGURAÇÕES EXPERIMENTAIS GLOBAIS
# Opções de warm-start para o Branch-and-Bound:
# "trivial", "nearest_neighbor", "convex_hull", "twice_around_the_tree", "christofides"
# ==============================================================================
WARM_START_STRATEGY: str = "trivial"
TIME_LIMIT_SECONDS: float = 1800.0  # 30 minutos
SAVE_INDIVIDUAL = False     # Caso falso, o programa apenas salva o resultado de todas as instâncias juntas

# Instâncias selecionadas (nome, sol_otima)
INSTANCES: dict[str, int] = {
    "burma14": 3323,
    "ulysses16": 6859,
    "gr17": 2085,
    "gr21": 2707,
    "ulysses22": 7013,
    "gr24": 1272,
    "bayg29": 1610,
    "bays29": 2020,
    "dantzig42": 699,
    "swiss42": 1273,
    "berlin52": 7542,
    "brazil58": 25395,
    "st70": 675
}

# Relativos à quantidade de testes feitos para medidas mais precisas de tempo (aplica apenas aos aproximativos)
VALID_TESTS = 100       # Quantidade de testes que serão validados
DISCARDED_TESTS = 20    # Quantidade de testes que serão descartados

# Tipos de testes que queremos fazer
TEST_TYPES: list[str] = [
    "branch-and-bound",         # Testar somente o branch_and_bound
    "twice-around-the-tree",    # Testar somente o twice-around-the-tree
    "christofides",             # Testar somente o Chirstofides
    "all"                       # Testar todos os algoritmos
]
TEST: str = TEST_TYPES[1]       # Mudar o número aqui pra mudar o teste


# Resolve uma única instância do TSP pelos três métodos e retorna um dicionário com todas as métricas.
def run_instance_benchmark(
    instance_name: str,
    optimal_cost: int,
    warm_start: str,
    time_limit: float,
    test_type: str
) -> dict:
    graph = load_tsp_instance(instance_name)
    print(f"\n=== Instância: {graph.name} (N={graph.dimension}, Tipo={graph.edge_weight_type}) ===")

    # 1. Algoritmo Aproximado: Twice-Around-the-Tree
    if test_type == "all" or test_type == "twice-around-the-tree":
        if VALID_TESTS == 1:
            tat_time, tat_factor, tat_prof = twice_around_the_tree(graph, optimal_cost, time_limit, debug=True)

        elif VALID_TESTS > 1:
            tat_time = 0.0
            peak_mem = 0
            time_mst = 0.0
            time_dfs = 0.0
            mem_mst = 0
            mem_dfs = 0

            for i in range(VALID_TESTS + DISCARDED_TESTS):
                t, tat_factor, p = twice_around_the_tree(graph, optimal_cost, time_limit, debug=False)
                print(f"Test number {i+1} time: {t:.6f}")
                if i >= DISCARDED_TESTS:
                    tat_time += t
                    peak_mem += p["total_peak_memory_bytes"]
                    time_mst += p["mst_construction"]["time_sec"]
                    time_dfs += p["preorder_shortcutting"]["time_sec"]
                    mem_mst += p["mst_construction"]["peak_memory_bytes"]
                    mem_dfs += p["preorder_shortcutting"]["peak_memory_bytes"]

            tat_time /= float(VALID_TESTS)
            peak_mem /= VALID_TESTS
            time_mst /= float(VALID_TESTS)
            time_dfs /= float(VALID_TESTS)
            mem_mst /= VALID_TESTS
            mem_dfs /= VALID_TESTS
            tat_prof = {
                "mst_construction": {
                    "time_sec": time_mst,
                    "peak_memory_bytes": int(mem_mst)
                },
                "preorder_shortcutting": {
                    "time_sec": time_dfs,
                    "peak_memory_bytes": int(mem_dfs)
                },
                "total_peak_memory_bytes": int(peak_mem)
            }

    # 2. Algoritmo Aproximado: Christofides
    if test_type == "all" or test_type == "christofides":
        if VALID_TESTS == 1:
            chr_time, chr_factor, chr_prof = christofides(graph, optimal_cost, time_limit, verbose=True)

        elif VALID_TESTS > 1:
            chr_time = 0.0
            peak_mem = 0
            time_mst = 0.0
            time_mat = 0.0
            time_cir = 0.0
            mem_mst = 0
            mem_mat = 0
            mem_cir = 0

            for i in range(VALID_TESTS + DISCARDED_TESTS):
                t, chr_factor, p = christofides(graph, optimal_cost, time_limit, verbose=False)
                print(f"Test number {i+1} time: {t}")
                if i >= DISCARDED_TESTS:
                    chr_time += t
                    peak_mem += p["total_peak_memory_bytes"]
                    time_mst += p["mst_construction"]["time_sec"]
                    time_mat += p["minimum_weight_perfect_matching"]["time_sec"]
                    time_cir += p["eulerian_circuit_and_shortcutting"]["time_sec"]
                    mem_mst += p["mst_construction"]["peak_memory_bytes"]
                    mem_mat += p["minimum_weight_perfect_matching"]["peak_memory_bytes"]
                    mem_cir += p["eulerian_circuit_and_shortcutting"]["peak_memory_bytes"]

            chr_time /= float(VALID_TESTS)
            peak_mem /= VALID_TESTS
            time_mst /= float(VALID_TESTS)
            time_mat /= float(VALID_TESTS)
            time_cir /= float(VALID_TESTS)
            mem_mst /= VALID_TESTS
            mem_mat /= VALID_TESTS
            mem_cir /= VALID_TESTS
            chr_prof = {
                "mst_construction": {
                    "time_sec": time_mst,
                    "peak_memory_bytes": int(mem_mst)
                },
                "minimum_weight_perfect_matching": {
                    "time_sec": time_mat,
                    "peak_memory_bytes": int(mem_mat)
                },
                "eulerian_circuit_and_shortcutting": {
                    "time_sec": time_cir,
                    "peak_memory_bytes": int(mem_cir)
                },
                "total_peak_memory_bytes": 0
            }

    # 3. Algoritmo Exato: Branch-and-Bound com warm-start selecionado
    if test_type == "all" or test_type == "branch-and-bound":
        bb_result = branch_and_bound(graph, warm_start=warm_start, time_limit=time_limit)
        status_str = "ÓTIMO" if bb_result["is_optimal"] else "TIMEOUT (30 min)"
        print(
            f"[Branch&Bound  | {graph.name}] Status: {status_str} | "
            f"Tempo Total: {bb_result['total_time_sec']:.6f}s | "
            f"Tempo Heurística ({warm_start}): {bb_result['heuristic_time_sec']:.6f}s | "
            f"Melhor Custo: {bb_result['best_cost']} (Ótimo={optimal_cost}) | "
            f"Expandidos: {bb_result['nodes_expanded']} | Podados: {bb_result['nodes_pruned']}"
        )

    results = {
        "instance_name": graph.name,
        "dimension": graph.dimension,
        "edge_weight_type": graph.edge_weight_type,
        "optimal_cost": optimal_cost,
    }

    if test_type == "all" or test_type == "branch-and-bound":
        results["branch_and_bound"] = bb_result

    if test_type == "all" or test_type == "twice-around-the-tree":
        results["twice-around-the-tree"] = {
            "total_time_sec": tat_time,
            "approx_factor": tat_factor,
            "subroutine_profiling": tat_prof,
        }

    if test_type == "all" or test_type == "christofides":
        results["christofides"] = {
            "total_time_sec": chr_time,
            "approx_factor": chr_factor,
            "subroutine_profiling": chr_prof,
        }

    return results


# Calcula métricas estatísticas de qualidade de ajuste (R² e RMSE) entre observado e predito.
def _compute_fit_metrics(y_obs: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_obs - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0
    rmse = float(math.sqrt(np.mean(residuals ** 2)))
    return {"r2": r2, "rmse": rmse}


# Ajusta curvas teóricas de complexidade temporal e espacial em função de N e calcula R² e RMSE.
def fit_complexity_curves(all_results: list[dict], test_type: str) -> dict:
    # Ordena os resultados por dimensão N crescente para o ajuste de regressão.
    ordered = sorted(all_results, key=lambda r: r["dimension"])
    n_all = np.array([r["dimension"] for r in ordered], dtype=np.float64)

    fits: dict[str, dict] = {}

    # Modelos paramétricos teóricos em função do número de vértices N.
    def exp_model(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(np.clip(b * n, -500.0, 500.0)) + c

    def quad_model(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * (n ** 2) + b * n + c

    def cubic_model(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * (n ** 3) + b * (n ** 2) + c

    # --- 1. Branch-and-Bound (apenas instâncias não censuradas pelo timeout para tempo) ---
    if test_type == "all" or test_type == "branch-and-bound":
        bb_valid = [r for r in ordered if not r["branch_and_bound"]["timed_out"]]
        if len(bb_valid) >= 3:
            n_bb = np.array([r["dimension"] for r in bb_valid], dtype=np.float64)
            t_bb = np.array([r["branch_and_bound"]["total_time_sec"] for r in bb_valid], dtype=np.float64)
            log_slope, log_intercept = np.polyfit(n_bb, np.log(np.maximum(t_bb, 1e-9)), 1)
            p0_exp = [max(math.exp(log_intercept), 1e-9), max(float(log_slope), 1e-3), 0.0]
            popt_t, _ = curve_fit(
                exp_model, n_bb, t_bb, p0=p0_exp, bounds=([0.0, 0.0, 0.0], [np.inf, 5.0, np.inf]), maxfev=10000
            )
            pred_t = exp_model(n_bb, *popt_t)
            fits["branch_and_bound_time"] = {
                "theoretical_complexity": "O(b^N) [Exponencial]",
                "formula": "a * exp(b * N) + c",
                "params": {"a": float(popt_t[0]), "b": float(popt_t[1]), "c": float(popt_t[2])},
                "n_values": n_bb.tolist(),
                "observed": t_bb.tolist(),
                "predicted": pred_t.tolist(),
                **_compute_fit_metrics(t_bb, pred_t),
            }

        s_bb = np.array([r["branch_and_bound"]["peak_memory_bytes"] for r in ordered], dtype=np.float64)
        if len(n_all) >= 3:
            popt_s, _ = curve_fit(quad_model, n_all, s_bb, bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
            pred_s = quad_model(n_all, *popt_s)
            fits["branch_and_bound_space"] = {
                "theoretical_complexity": "O(N^2) [Matriz de distâncias + pilha DFS O(N)]",
                "formula": "a * N^2 + b * N + c",
                "params": {"a": float(popt_s[0]), "b": float(popt_s[1]), "c": float(popt_s[2])},
                "n_values": n_all.tolist(),
                "observed": s_bb.tolist(),
                "predicted": pred_s.tolist(),
                **_compute_fit_metrics(s_bb, pred_s),
            }

    # --- 2. Twice-Around-the-Tree: Tempo O(N^2) e Espaço O(N^2) ---
    if test_type == "all" or test_type == "twice-around-the-tree":
        t_tat = np.array([r["twice_around_the_tree"]["total_time_sec"] for r in ordered], dtype=np.float64)
        s_tat = np.array(
            [r["twice_around_the_tree"]["subroutine_profiling"]["total_peak_memory_bytes"] for r in ordered],
            dtype=np.float64,
        )
        if len(n_all) >= 3:
            popt_t_tat, _ = curve_fit(quad_model, n_all, t_tat, bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
            pred_t_tat = quad_model(n_all, *popt_t_tat)
            fits["twice_around_time"] = {
                "theoretical_complexity": "O(N^2) [Prim MST em grafo completo + DFS]",
                "formula": "a * N^2 + b * N + c",
                "params": {"a": float(popt_t_tat[0]), "b": float(popt_t_tat[1]), "c": float(popt_t_tat[2])},
                "n_values": n_all.tolist(),
                "observed": t_tat.tolist(),
                "predicted": pred_t_tat.tolist(),
                **_compute_fit_metrics(t_tat, pred_t_tat),
            }

            popt_s_tat, _ = curve_fit(quad_model, n_all, s_tat, bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
            pred_s_tat = quad_model(n_all, *popt_s_tat)
            fits["twice_around_space"] = {
                "theoretical_complexity": "O(N^2) [Grafo completo e MST]",
                "formula": "a * N^2 + b * N + c",
                "params": {"a": float(popt_s_tat[0]), "b": float(popt_s_tat[1]), "c": float(popt_s_tat[2])},
                "n_values": n_all.tolist(),
                "observed": s_tat.tolist(),
                "predicted": pred_s_tat.tolist(),
                **_compute_fit_metrics(s_tat, pred_s_tat),
            }

    # --- 3. Christofides: Tempo O(N^3) [Blossom MWPM] e Espaço O(N^2) ---
    if test_type == "all" or test_type == "christofides":
        t_chr = np.array([r["christofides"]["total_time_sec"] for r in ordered], dtype=np.float64)
        s_chr = np.array(
            [r["christofides"]["subroutine_profiling"]["total_peak_memory_bytes"] for r in ordered],
            dtype=np.float64,
        )
        if len(n_all) >= 3:
            popt_t_chr, _ = curve_fit(cubic_model, n_all, t_chr, bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
            pred_t_chr = cubic_model(n_all, *popt_t_chr)
            fits["christofides_time"] = {
                "theoretical_complexity": "O(N^3) [Emparelhamento Perfeito de Peso Mínimo - Blossom]",
                "formula": "a * N^3 + b * N^2 + c",
                "params": {"a": float(popt_t_chr[0]), "b": float(popt_t_chr[1]), "c": float(popt_t_chr[2])},
                "n_values": n_all.tolist(),
                "observed": t_chr.tolist(),
                "predicted": pred_t_chr.tolist(),
                **_compute_fit_metrics(t_chr, pred_t_chr),
            }

            popt_s_chr, _ = curve_fit(quad_model, n_all, s_chr, bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
            pred_s_chr = quad_model(n_all, *popt_s_chr)
            fits["christofides_space"] = {
                "theoretical_complexity": "O(N^2) [Subgrafo induzido dos vértices ímpares]",
                "formula": "a * N^2 + b * N + c",
                "params": {"a": float(popt_s_chr[0]), "b": float(popt_s_chr[1]), "c": float(popt_s_chr[2])},
                "n_values": n_all.tolist(),
                "observed": s_chr.tolist(),
                "predicted": pred_s_chr.tolist(),
                **_compute_fit_metrics(s_chr, pred_s_chr),
            }

    return fits


# Orquestra a execução experimental em todas as instâncias da TSPLIB e salva os resultados em JSON.
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results" / f"warm_start_{WARM_START_STRATEGY}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for instance_name, optimal_cost in INSTANCES.items():
        result = run_instance_benchmark(
            instance_name=instance_name,
            optimal_cost=optimal_cost,
            warm_start=WARM_START_STRATEGY,
            time_limit=TIME_LIMIT_SECONDS,
            test_type=TEST
        )
        all_results.append(result)

        # Persiste individualmente os dados de cada instância logo após a sua conclusão.
        if SAVE_INDIVIDUAL:
            instance_file = results_dir / f"{instance_name}.json"
            with instance_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    # Salva o consolidado geral das execuções.
    summary_file = results_dir / "all_instances_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Ajusta as curvas teóricas de complexidade temporal e espacial e salva métricas (R² e RMSE).
    if TEST == "all":
        curve_fits = fit_complexity_curves(all_results, TEST)
        fits_file = results_dir / "complexity_curve_fits.json"
        with fits_file.open("w", encoding="utf-8") as f:
            json.dump(curve_fits, f, indent=2, ensure_ascii=False)

        print("\n=== Resumo do Ajuste de Curvas Teóricas (R² e RMSE) ===")
        for model_name, info in curve_fits.items():
            print(
                f"- {model_name:24s} | {info['theoretical_complexity']} | "
                f"R² = {info['r2']:.4f} | RMSE = {info['rmse']:.6e}"
            )


if __name__ == "__main__":
    main()