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

INSTANCES: dict[str, int] = {
    "bayg29": 1610,
    "bays29": 2020,
    "berlin52": 7542,
    "brazil58": 25395,
    "burma14": 3323,
    "dantzig42": 699,
    "gr17": 2085,
    "gr21": 2707,
    "gr24": 1272,
    "st70": 675,
    "swiss42": 1273,
    "ulysses16": 6859,
    "ulysses22": 7013,
}


# Resolve uma única instância do TSP pelos três métodos e retorna um dicionário com todas as métricas.
def run_instance_benchmark(
    instance_name: str,
    optimal_cost: int,
    warm_start: str,
    time_limit: float,
) -> dict:
    graph = load_tsp_instance(instance_name)
    print(f"\n=== Instância: {graph.name} (N={graph.dimension}, Tipo={graph.edge_weight_type}) ===")

    # 1. Algoritmo Aproximado: Twice-Around-the-Tree
    tat_time, tat_factor, tat_prof = twice_around_the_tree(
        graph, optimal_cost=optimal_cost, time_limit=time_limit, verbose=True
    )

    # 2. Algoritmo Aproximado: Christofides
    chr_time, chr_factor, chr_prof = christofides(
        graph, optimal_cost=optimal_cost, time_limit=time_limit, verbose=True
    )

    # 3. Algoritmo Exato: Branch-and-Bound com warm-start selecionado
    bb_result = branch_and_bound(
        graph, warm_start=warm_start, time_limit=time_limit
    )
    status_str = "ÓTIMO" if bb_result["is_optimal"] else "TIMEOUT (30 min)"
    print(
        f"[Branch&Bound  | {graph.name}] Status: {status_str} | "
        f"Tempo Total: {bb_result['total_time_sec']:.6f}s | "
        f"Tempo Heurística ({warm_start}): {bb_result['heuristic_time_sec']:.6f}s | "
        f"Melhor Custo: {bb_result['best_cost']} (Ótimo={optimal_cost}) | "
        f"Expandidos: {bb_result['nodes_expanded']} | Podados: {bb_result['nodes_pruned']}"
    )

    return {
        "instance_name": graph.name,
        "dimension": graph.dimension,
        "edge_weight_type": graph.edge_weight_type,
        "optimal_cost": optimal_cost,
        "branch_and_bound": bb_result,
        "twice_around_the_tree": {
            "total_time_sec": tat_time,
            "approx_factor": tat_factor,
            "subroutine_profiling": tat_prof,
        },
        "christofides": {
            "total_time_sec": chr_time,
            "approx_factor": chr_factor,
            "subroutine_profiling": chr_prof,
        },
    }


# Calcula métricas estatísticas de qualidade de ajuste (R² e RMSE) entre observado e predito.
def _compute_fit_metrics(y_obs: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_obs - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0
    rmse = float(math.sqrt(np.mean(residuals ** 2)))
    return {"r2": r2, "rmse": rmse}


# Ajusta curvas teóricas de complexidade temporal e espacial em função de N e calcula R² e RMSE.
def fit_complexity_curves(all_results: list[dict]) -> dict:
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

    # Ordena a execução da menor para a maior dimensão N para obter resultados parciais rapidamente.
    sorted_instances = sorted(INSTANCES.items(), key=lambda item: int("".join(filter(str.isdigit, item[0]))))

    for instance_name, optimal_cost in sorted_instances:
        result = run_instance_benchmark(
            instance_name=instance_name,
            optimal_cost=optimal_cost,
            warm_start=WARM_START_STRATEGY,
            time_limit=TIME_LIMIT_SECONDS,
        )
        all_results.append(result)

        # Persiste individualmente os dados de cada instância logo após a sua conclusão.
        instance_file = results_dir / f"{instance_name}.json"
        with instance_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Salva o consolidado geral das execuções.
    summary_file = results_dir / "all_instances_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Ajusta as curvas teóricas de complexidade temporal e espacial e salva métricas (R² e RMSE).
    curve_fits = fit_complexity_curves(all_results)
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