# ===================================================================================== #
# ESTA PARTE FOI ARQUITETADA E RODADA NO COLAB COM OS JSON GERADOS PELO MAIN NO CONTENT # 
# ===================================================================================== #

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from IPython.display import display

# ============================================================
# 1. BUSCA AUTOMÁTICA DOS JSON NA PASTA /content
# ============================================================

import glob
import os
import json

json_data = {}

json_files = glob.glob("/content/**/*.json", recursive=True)

print(f"JSONs encontrados: {len(json_files)}")

for filepath in json_files:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_data[filepath] = data

        print(f"Carregado: {os.path.basename(filepath)}")

    except Exception as e:
        print(f"Erro em {filepath}: {e}")

print(f"\nTotal carregado: {len(json_data)} arquivos")

# ============================================================
# 2. EXTRAÇÃO DOS DADOS
# ============================================================

records = []

for filename, data in json_data.items():

    # Ignora JSONs cuja raiz não seja um dicionário
    if not isinstance(data, dict):
        print(f"Ignorado (formato incompatível): {os.path.basename(filename)}")
        continue

    # Verifica se é um JSON de resultados do TSP
    strategies = data.get("branch_and_bound", {})

    if not isinstance(strategies, dict) or not strategies:
        print(f"Ignorado (sem branch_and_bound): {os.path.basename(filename)}")
        continue

    instance_name = data.get(
        "instance_name",
        os.path.splitext(os.path.basename(filename))[0]
    )

    dimension = data.get("dimension")
    optimal_cost = data.get("optimal_cost")

    for strategy_name, result in strategies.items():

        if not isinstance(result, dict):
            continue

        row = {
            "instance": instance_name,
            "dimension": dimension,
            "optimal_cost": optimal_cost,
            "strategy": strategy_name,

            # Qualidade da solução
            "best_cost": result.get("best_cost"),
            "is_optimal": result.get("is_optimal"),
            "timed_out": result.get("timed_out"),

            # Tempos
            "total_time_sec": result.get("total_time_sec"),
            "heuristic_time_sec": result.get("heuristic_time_sec"),

            # Memória
            "peak_memory_bytes": result.get("peak_memory_bytes"),
            "heuristic_peak_memory_bytes":
                result.get("heuristic_peak_memory_bytes"),

            # Estatísticas da busca
            "nodes_expanded": result.get("nodes_expanded"),
            "nodes_pruned": result.get("nodes_pruned"),
            "candidates_evaluated": result.get("candidates_evaluated"),

            # Estatísticas das heurísticas
            "heuristic_nodes_visited":
                result.get("heuristic_nodes_visited"),
            "heuristic_nodes_pruned":
                result.get("heuristic_nodes_pruned"),
        }

        records.append(row)

# Cria o DataFrame
df = pd.DataFrame(records)

if df.empty:
    print("\nNenhum resultado TSP válido foi encontrado.")

else:
    print(f"\nInstâncias TSP processadas: {df['instance'].nunique()}")
    print(f"Total de resultados: {len(df)}")

    display(df)

# ============================================================
# 3. DATAFRAME
# ============================================================

df = pd.DataFrame(records)

if df.empty:

    print("\nNenhum resultado foi encontrado.")
    print("Confira se os JSONs possuem a chave 'branch_and_bound'.")

else:

    # Converte valores numéricos
    numeric_columns = [
        "dimension",
        "optimal_cost",
        "best_cost",
        "total_time_sec",
        "heuristic_time_sec",
        "peak_memory_bytes",
        "heuristic_peak_memory_bytes",
        "nodes_expanded",
        "nodes_pruned",
        "candidates_evaluated",
        "heuristic_nodes_visited",
        "heuristic_nodes_pruned"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # Conversão de memória: bytes -> MB
    # --------------------------------------------------------

    df["peak_memory_mb"] = (
        df["peak_memory_bytes"] / (1024 ** 2)
    )

    df["heuristic_peak_memory_mb"] = (
        df["heuristic_peak_memory_bytes"] / (1024 ** 2)
    )

    # --------------------------------------------------------
    # Gap percentual em relação ao ótimo
    # --------------------------------------------------------

    df["gap_percent"] = np.where(
        (df["optimal_cost"] > 0) &
        df["best_cost"].notna(),

        100 * (df["best_cost"] - df["optimal_cost"])
        / df["optimal_cost"],

        np.nan
    )

    # --------------------------------------------------------
    # Fator best_cost / optimal_cost
    # --------------------------------------------------------

    df["cost_ratio"] = np.where(
        (df["optimal_cost"] > 0) &
        df["best_cost"].notna(),

        df["best_cost"] / df["optimal_cost"],

        np.nan
    )

    print("\nDados carregados:")
    display(df)

    print("\nInstâncias:")
    print(df["instance"].unique())

    print("\nEstratégias:")
    print(df["strategy"].unique())

# ============================================================
# 4. FUNÇÃO PARA GERAR GRÁFICOS COMPARATIVOS
# ============================================================

def plot_bar_metric(df, metric, title, ylabel, log=False):

    plot_df = df.dropna(subset=[metric])

    if plot_df.empty:
        print(f"Métrica indisponível: {metric}")
        return

    pivot = plot_df.pivot_table(
        index="instance",
        columns="strategy",
        values=metric,
        aggfunc="mean"
    )

    cores = [
        "#FF0000",  # Vermelho
        "#0056FF",   # Azul
        "#20A900",   # Verde
        "#B800FF",   # Roxo
        "#FF8700",   # Laranja
    ]

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8,
        color=cores
    )
    if log:
      ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Instância")
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Warm Start", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. GRÁFICO: CUSTO DA SOLUÇÃO
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "best_cost",
        "Custo da melhor solução encontrada",
        "Custo",
        log=True
    )

# ============================================================
# 6. GRÁFICO: GAP PERCENTUAL EM RELAÇÃO AO ÓTIMO
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "gap_percent",
        "Gap percentual em relação à solução ótima",
        "Gap (%)",
        log=True
    )
# ============================================================
# 7. GRÁFICO: TEMPO TOTAL
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "total_time_sec",
        "Tempo total de execução do Branch-and-Bound",
        "Tempo (s)"
    )

# ============================================================
# 8. GRÁFICO: TEMPO DO WARM START
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "heuristic_time_sec",
        "Tempo de execução da heurística de warm start",
        "Tempo (s)",
        log=True
    )

# ============================================================
# 9. GRÁFICO: MEMÓRIA DE PICO DO B&B
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "peak_memory_mb",
        "Memória de pico do Branch-and-Bound",
        "Memória (MB)"
    )

# ============================================================
# 10. GRÁFICO: MEMÓRIA DE PICO DO WARM START
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "heuristic_peak_memory_mb",
        "Memória de pico da heurística de warm start",
        "Memória (MB)",
        log=True
    )

# ============================================================
# 11. GRÁFICO: NÓS EXPANDIDOS
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "nodes_expanded",
        "Número de nós expandidos",
        "Nós expandidos"
    )

# ============================================================
# 12. GRÁFICO: NÓS PODADOS
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "nodes_pruned",
        "Número de nós podados",
        "Nós podados"
    )

# ============================================================
# 13. GRÁFICO: CANDIDATOS AVALIADOS
# ============================================================

if not df.empty:

    plot_bar_metric(
        df,
        "candidates_evaluated",
        "Número de candidatos avaliados",
        "Candidatos"
    )


# ============================================================
# 14. TABELA RESUMO POR WARM START
# ============================================================

if not df.empty:

    summary_columns = [
        "best_cost",
        "gap_percent",
        "cost_ratio",
        "total_time_sec",
        "heuristic_time_sec",
        "peak_memory_mb",
        "nodes_expanded",
        "nodes_pruned",
        "candidates_evaluated"
    ]

    available_columns = [
        col for col in summary_columns
        if col in df.columns
    ]

    summary = df.groupby("strategy")[available_columns].mean()

    print("\n" + "=" * 70)
    print("MÉDIAS POR ESTRATÉGIA")
    print("=" * 70)

    display(summary.round(4))

# ============================================================
# 15. TABELA: EXECUÇÕES QUE ATINGIRAM TIMEOUT
# ============================================================

if not df.empty:

    print("\nExecuções que atingiram timeout:")

    timeouts = df[df["timed_out"] == True][
        ["instance", "strategy", "total_time_sec", "best_cost"]
    ]

    display(timeouts)

# ------------------------------------------------------------
# 16. BUSCA E LEITURA DO ARQUIVO DE CARACTERISTICAS
# ------------------------------------------------------------

characteristic_files = glob.glob(
    "/content/**/*",
    recursive=True
)

# Procura arquivo chamado "caracteristica", com ou sem extensão
characteristic_files = [
    filepath for filepath in characteristic_files
    if os.path.isfile(filepath)
    and os.path.basename(filepath).lower().startswith("caracteristica")
]

if not characteristic_files:
    raise FileNotFoundError(
        "Arquivo de características não encontrado no /content."
    )

characteristic_records = []

for filepath in characteristic_files:

    try:

        # Lê o arquivo como CSV
        temp_df = pd.read_csv(filepath)

        # Remove espaços extras dos nomes das colunas
        temp_df.columns = temp_df.columns.str.strip()

        if "instance" not in temp_df.columns:
            print(f"Ignorado (sem coluna instance): {filepath}")
            continue

        characteristic_records.append(temp_df)

        print(f"Características carregadas: {filepath}")

    except Exception as e:

        print(f"Erro ao ler {filepath}: {e}")


# Junta os arquivos encontrados
if characteristic_records:

    characteristics = pd.concat(
        characteristic_records,
        ignore_index=True
    )

else:

    raise ValueError(
        "Nenhum arquivo de características válido foi carregado."
    )


print("\nTotal de características carregadas:", len(characteristics))

display(characteristics)


# ------------------------------------------------------------
# 17. PADRONIZACAO DOS NOMES DAS INSTANCIAS
# ------------------------------------------------------------

df["instance"] = (
    df["instance"]
    .astype(str)
    .str.strip()
    .str.lower()
)

characteristics["instance"] = (
    characteristics["instance"]
    .astype(str)
    .str.strip()
    .str.lower()
)

characteristics = characteristics.drop_duplicates(
    subset="instance"
)


# ------------------------------------------------------------
# 18. CRUZAMENTO DOS DATAFRAMES
# ------------------------------------------------------------

df_structural = df.merge(
    characteristics,
    on="instance",
    how="inner"
)

matched_instances = df_structural["instance"].unique()

unmatched_instances = sorted(
    set(df["instance"]) - set(characteristics["instance"])
)

print("\nInstâncias cruzadas:", len(matched_instances))
print("Resultados cruzados:", len(df_structural))

if unmatched_instances:

    print("\nInstâncias de desempenho sem características:")
    print(unmatched_instances)

display(df_structural.head())

# --------------------------------------------------------
# 19. CONFIGURACAO
# --------------------------------------------------------

warm_start_colors = {
    "trivial": "#FF0000",
    "nearest_neighbor": "#0056FF",
    "convex_hull": "#20A900",
    "twice_around_the_tree": "#B800FF",
    "christofides": "#FF8700"
}


# --------------------------------------------------------
# 20. FUNCAO DE DISPERSAO ESTRUTURA x DESEMPENHO
# --------------------------------------------------------

def plot_structural_scatter(
    x,
    y,
    xlabel,
    ylabel,
    title,
    log_y=False
):

    plot_df = df_structural.dropna(
        subset=[x, y, "strategy"]
    )

    if plot_df.empty:
        print(f"Dados insuficientes para: {title}")
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    for strategy in plot_df["strategy"].unique():

        sub = plot_df[
            plot_df["strategy"] == strategy
        ]

        ax.scatter(
            sub[x],
            sub[y],
            label=strategy,
            color=warm_start_colors.get(strategy),
            s=75,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.4
        )

    if log_y:
        # Log não aceita valores <= 0
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        title="Warm Start",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------
# 21. BOX PLOT: GAP POR CLASSE DE ASPECT RATIO
# --------------------------------------------------------
def plot_box_by_class(
        class_column,
        metric,
        xlabel,
        ylabel,
        title
    ):

    if class_column not in df_structural.columns:
        print(f"Coluna ausente: {class_column}")
        return

    classes = ["Low", "Medium", "High"]

    plot_data = [
            df_structural.loc[
                df_structural[class_column] == cls,
                metric
            ].dropna()
            for cls in classes
        ]

        # Remove categorias sem dados
    valid_data = [
            (cls, values)
            for cls, values in zip(classes, plot_data)
            if len(values) > 0
        ]

    if not valid_data:
        print(f"Sem dados para: {title}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(
            [values for _, values in valid_data],
            labels=[cls for cls, _ in valid_data],
            showmeans=True
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.show()


plot_box_by_class(
        "aspect_ratio_class",
        "gap_percent",
        "Classe de Aspect Ratio",
        "Gap (%)",
        "Gap por classe de Aspect Ratio"
    )


# --------------------------------------------------------
# 22. BOX PLOT: GAP POR CLASSE DE ENTROPIA
# --------------------------------------------------------

plot_box_by_class(
        "spatial_entropy_class",
        "gap_percent",
        "Classe de Entropia Espacial",
        "Gap (%)",
        "Gap por classe de Entropia Espacial"
    )


# --------------------------------------------------------
# 23. BOX PLOT: NOS EXPANDIDOS POR CLASSE MST
# --------------------------------------------------------

plot_box_by_class(
        "mst_max_degree_class",
        "nodes_expanded",
        "Classe de Grau Máximo da MST",
        "Nós expandidos",
        "Nós expandidos por classe de Grau Máximo da MST"
    )


# --------------------------------------------------------
# 24. BOXPLOT: TEMPO POR CLASSE DE HULL RATIO
# --------------------------------------------------------

plot_box_by_class(
        "hull_point_ratio_class",
        "total_time_sec",
        "Classe de Proporção do Fecho Convexo",
        "Tempo (s)",
        "Tempo de execução por classe de Fecho Convexo"
    )


# --------------------------------------------------------
# 25. CORRELAÇÃO ENTRE ESTRUTURA E DESEMPENHO
# --------------------------------------------------------

structural_metrics = [
    "dimension",
    "aspect_ratio",
    "spatial_entropy",
    "cluster_count",
    "mst_max_degree",
]

performance_metrics = [
    "total_time_sec",
    "nodes_expanded",
    "nodes_pruned",
    "candidates_evaluated"
]

# Mantém apenas colunas existentes
structural_columns = [
    col for col in structural_metrics
    if col in df_structural.columns
]

performance_columns = [
    col for col in performance_metrics
    if col in df_structural.columns
]

# Cada instância aparece 5 vezes (warm starts).
# Agregamos pela média para evitar duplicação.
df_instance_avg = (
    df_structural
    .groupby("instance", as_index=False)
    .mean(numeric_only=True)
)

# Calcula SOMENTE correlação estrutural vs desempenho
correlation_matrix = df_instance_avg[
    structural_columns + performance_columns
].corr().loc[
    structural_columns,
    performance_columns
]

# --------------------------------------------------------
# GRÁFICO
# --------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))

im = ax.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

# Eixos
ax.set_xticks(range(len(performance_columns)))
ax.set_yticks(range(len(structural_columns)))

ax.set_xticklabels(performance_columns, rotation=45, ha="right")
ax.set_yticklabels(structural_columns)

# Valores nas células
for i in range(len(structural_columns)):
    for j in range(len(performance_columns)):

        value = correlation_matrix.iloc[i, j]

        if pd.notna(value):
            ax.text(
                j, i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

plt.colorbar(im, ax=ax, label="Correlação")

ax.set_title(
    "Correlação entre Características Estruturais e Desempenho"
)

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 26. EXPORTAR DATAFRAME CRUZADO
# --------------------------------------------------------

df_structural.to_csv(
    "/content/results_structural_analysis.csv",
    index=False
)

print(
    "\nCSV estrutural cruzado salvo em:"
    "\n/content/results_structural_analysis.csv"
)

# ==============================================================
# REGRESSÃO EMPÍRICA DE COMPLEXIDADE - CHRISTOFIDES
# ==============================================================

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# LOCALIZAÇÃO E CARREGAMENTO DOS JSONS DO CHRISTOFIDES
# ==============================================================

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

json_files = glob.glob("/content/*_c.json")

print(f"JSONs encontrados: {len(json_files)}")

data = []

for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        instance = json.load(f)

    profiling = instance["christofides"]["subroutine_profiling"]

    data.append({
        "instance": instance["instance_name"],
        "n": instance["dimension"],
        "time": instance["christofides"]["total_time_sec"],
        "space": profiling["total_peak_memory_bytes"]
    })

data.sort(key=lambda x: x["n"])

if not data:
    raise ValueError("Nenhum JSON do Christofides foi encontrado em /content.")

n_values = np.array([d["n"] for d in data], dtype=float)
time_values = np.array([d["time"] for d in data], dtype=float)
space_values = np.array([d["space"] for d in data], dtype=float)

print(f"Instâncias carregadas: {len(data)}")

for d in data:
    print(
        f"{d['instance']:12} | n = {d['n']:3} | "
        f"Tempo = {d['time']:.6f} s | "
        f"Memória = {d['space']} bytes"
    )
# --------------------------------------------------------------
# 2. CARREGAMENTO DOS DADOS
# --------------------------------------------------------------

data = []

for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        instance = json.load(f)

    christofides = instance["christofides"]
    profiling = christofides["subroutine_profiling"]

    data.append({
        "instance": instance["instance_name"],
        "n": instance["dimension"],
        "time": christofides["total_time_sec"],
        "space": profiling["total_peak_memory_bytes"]
    })

data.sort(key=lambda x: x["n"])

if not data:
    raise ValueError("Nenhum JSON do Christofides foi encontrado.")

n_values = np.array([d["n"] for d in data], dtype=float)
time_values = np.array([d["time"] for d in data], dtype=float)
space_values = np.array([d["space"] for d in data], dtype=float)

print(f"Instâncias carregadas: {len(data)}")

for d in data:
    print(
        f"{d['instance']:12} | n = {d['n']:3} | "
        f"Tempo = {d['time']:.6f} s | "
        f"Memória = {d['space']} bytes"
    )

# --------------------------------------------------------------
# 3. MODELOS DE COMPLEXIDADE
# --------------------------------------------------------------

def complexity_functions(n):
    return {
        "Constante O(1)": np.ones_like(n),
        "Logarítmica O(log n)": np.log(n),
        "Linear O(n)": n,
        "Linear-log O(n log n)": n * np.log(n),
        "Quadrática O(n²)": n**2,
        "Cúbica O(n³)": n**3,
    }

# --------------------------------------------------------------
# 4. REGRESSÃO E SELEÇÃO PELO R²
# --------------------------------------------------------------

def fit_complexity_models(n, y):

    models = complexity_functions(n)
    results = {}

    ss_total = np.sum((y - np.mean(y)) ** 2)

    for name, func in models.items():

        # Modelo: y = a*f(n) + b
        X = np.column_stack((func, np.ones_like(n)))

        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)

        predictions = X @ coefficients

        ss_residual = np.sum((y - predictions) ** 2)

        if ss_total == 0:
            r2 = 1.0 if ss_residual == 0 else 0.0
        else:
            r2 = 1 - ss_residual / ss_total

        results[name] = {
            "coefficients": coefficients,
            "r2": r2,
            "predictions": predictions
        }

    best_model = max(results, key=lambda name: results[name]["r2"])

    return results, best_model

# --------------------------------------------------------------
# 5. GRÁFICO DE REGRESSÃO
# --------------------------------------------------------------

def plot_complexity_regression(n, y, metric_name, unit):

    results, best_model = fit_complexity_models(n, y)

    # Curva suave para exibição
    n_smooth = np.linspace(min(n), max(n), 300)

    plt.figure(figsize=(10, 6))

    # Dados observados
    plt.scatter(
        n, y,
        s=65,
        label="Dados observados",
        zorder=3
    )

    # Melhor modelo
    best = results[best_model]
    a, b = best["coefficients"]

    func = complexity_functions(n_smooth)[best_model]
    y_smooth = a * func + b

    plt.plot(
        n_smooth,
        y_smooth,
        linewidth=2.5,
        label=f"Melhor ajuste: {best_model} (R² = {best['r2']:.4f})"
    )

    plt.title(f"Regressão empírica de complexidade - {metric_name}")
    plt.xlabel("Número de cidades (n)")
    plt.ylabel(unit)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Exibe ranking dos modelos
    print(f"\n--- Modelos para {metric_name} ---")

    sorted_models = sorted(
        results.items(),
        key=lambda item: item[1]["r2"],
        reverse=True
    )

    for name, result in sorted_models:
        print(f"{name:25} | R² = {result['r2']:.6f}")

    print(f"\nMelhor modelo: {best_model}")

    return results, best_model

# --------------------------------------------------------------
# 6. REGRESSÃO DE TEMPO
# --------------------------------------------------------------

time_results, best_time_model = plot_complexity_regression(
    n_values,
    time_values,
    "Tempo",
    "Tempo (segundos)"
)

# --------------------------------------------------------------
# 7. REGRESSÃO DE ESPAÇO
# --------------------------------------------------------------

space_results, best_space_model = plot_complexity_regression(
    n_values,
    space_values,
    "Espaço",
    "Pico de memória (bytes)"
)
