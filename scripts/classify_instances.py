# ============================================================
# CLASSIFICADOR ESTRUTURAL DE INSTÂNCIAS TSP
# Aspect Ratio | Convex Hull | Entropia | DBSCAN | MST
# ============================================================

# Dependências (execute uma vez no terminal, se necessário):
# pip install numpy scipy matplotlib scikit-learn pandas

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull, distance_matrix, QhullError
from sklearn.cluster import DBSCAN


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

from pathlib import Path

# Diretório onde está o classify_instances.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Raiz do projeto (um nível acima de fixed_scripts)
PROJECT_ROOT = SCRIPT_DIR.parent

# Instâncias TSP
INSTANCES_DIR = PROJECT_ROOT / "data"

# Saída dos resultados
OUTPUT_DIR = PROJECT_ROOT / "results" / "instance_characteristics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZE = 10


# ============================================================
# 2. LEITURA DOS ARQUIVOS TSP
# ============================================================

def read_tsp_coordinates(filepath):

    coordinates = []
    reading_coordinates = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            line = line.strip()

            if line == "NODE_COORD_SECTION":
                reading_coordinates = True
                continue

            if line == "EOF":
                break

            if reading_coordinates:

                parts = line.split()

                if len(parts) >= 3:

                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coordinates.append([x, y])

                    except ValueError:
                        continue

    coordinates = np.array(coordinates, dtype=float)

    if len(coordinates) == 0:
        raise ValueError("Nenhuma coordenada encontrada.")

    return coordinates


# ============================================================
# 3. ASPECT RATIO DA BOUNDING BOX
# ============================================================

def compute_aspect_ratio(points):

    x_range = np.ptp(points[:, 0])
    y_range = np.ptp(points[:, 1])

    smaller = min(x_range, y_range)
    larger = max(x_range, y_range)

    if smaller == 0:
        return np.inf

    return larger / smaller


# ============================================================
# 4. FECHO CONVEXO
# ============================================================

def compute_hull_ratio(points):

    n = len(points)

    if n < 3:
        return 1.0, n

    try:
        hull = ConvexHull(points)

        hull_points = len(hull.vertices)
        hull_ratio = hull_points / n

        return hull_ratio, hull_points

    except QhullError:
        return np.nan, 0


# ============================================================
# 5. ENTROPIA ESPACIAL
# ============================================================

def compute_spatial_entropy(points, grid_size=GRID_SIZE):

    x = points[:, 0]
    y = points[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Evita divisão por zero em eixos sem extensão
    if x_max == x_min:
        x_max += 1e-10

    if y_max == y_min:
        y_max += 1e-10

    # Normalização para [0,1]
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)

    x_bins = np.minimum(
        (x_norm * grid_size).astype(int),
        grid_size - 1
    )

    y_bins = np.minimum(
        (y_norm * grid_size).astype(int),
        grid_size - 1
    )

    # Distribuição de ocupação das células
    cells = list(zip(x_bins, y_bins))

    _, counts = np.unique(cells, return_counts=True)

    probabilities = counts / len(points)

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    # Entropia normalizada
    max_entropy = np.log(grid_size ** 2)

    normalized_entropy = entropy / max_entropy

    return normalized_entropy


# ============================================================
# 6. CLUSTERING - DBSCAN
# ============================================================

def compute_clustering(points):

    n = len(points)

    if n < 3:

        return {
            "cluster_count": 0,
            "clustered_point_ratio": 0.0,
            "noise_ratio": 1.0
        }

    distances = distance_matrix(points, points)

    # Distância do vizinho mais próximo de cada ponto
    np.fill_diagonal(distances, np.inf)

    nearest_distances = distances.min(axis=1)

    median_nearest_distance = np.median(nearest_distances)

    eps = 2.5 * median_nearest_distance

    if eps <= 0:
        eps = 1e-10

    dbscan = DBSCAN(
        eps=eps,
        min_samples=3
    )

    labels = dbscan.fit_predict(points)

    # -1 representa ruído
    noise_mask = labels == -1

    clustered_mask = ~noise_mask

    cluster_labels = set(labels)
    cluster_labels.discard(-1)

    cluster_count = len(cluster_labels)

    clustered_point_ratio = (
        np.sum(clustered_mask) / n
    )

    noise_ratio = (
        np.sum(noise_mask) / n
    )

    return {
        "cluster_count": cluster_count,
        "clustered_point_ratio": clustered_point_ratio,
        "noise_ratio": noise_ratio
    }


# ============================================================
# 7. MST - ÁRVORE GERADORA MÍNIMA (PRIM)
# ============================================================

def compute_mst_statistics(points):

    n = len(points)

    if n < 2:

        return {
            "mst_max_degree": 0,
            "mst_mean_degree": 0,
            "mst_degree_std": 0,
            "mst_leaf_count": 0,
            "mst_leaf_ratio": 0
        }

    distances = distance_matrix(points, points)

    visited = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    parent = np.full(n, -1)

    min_dist[0] = 0

    degrees = np.zeros(n, dtype=int)

    # Prim
    for _ in range(n):

        candidates = np.where(~visited)[0]

        u = candidates[np.argmin(min_dist[candidates])]

        visited[u] = True

        if parent[u] != -1:

            degrees[u] += 1
            degrees[parent[u]] += 1

        for v in range(n):

            if not visited[v] and distances[u, v] < min_dist[v]:

                min_dist[v] = distances[u, v]
                parent[v] = u

    max_degree = degrees.max()
    mean_degree = degrees.mean()
    degree_std = degrees.std()

    leaf_count = np.sum(degrees == 1)
    leaf_ratio = leaf_count / n

    return {
        "mst_max_degree": max_degree,
        "mst_mean_degree": mean_degree,
        "mst_degree_std": degree_std,
        "mst_leaf_count": leaf_count,
        "mst_leaf_ratio": leaf_ratio
    }


# ============================================================
# 8. ANALISE COMPLETA DE UMA INSTÂNCIA
# ============================================================

def analyze_instance(filepath):

    points = read_tsp_coordinates(filepath)

    aspect_ratio = compute_aspect_ratio(points)

    hull_ratio, hull_points = compute_hull_ratio(points)

    spatial_entropy = compute_spatial_entropy(points)

    clustering = compute_clustering(points)

    mst_stats = compute_mst_statistics(points)

    result = {
        "instance": filepath.stem,
        "dimension": len(points),
        "aspect_ratio": aspect_ratio,
        "hull_point_count": hull_points,
        "hull_point_ratio": hull_ratio,
        "spatial_entropy": spatial_entropy,
        **clustering,
        **mst_stats
    }

    return result


# ============================================================
# 9. CLASSIFICAÇÃO LOW / MEDIUM / HIGH
# ============================================================

def classify_metric(series):

    valid = series.dropna()

    result = pd.Series(
        index=series.index,
        dtype="object"
    )

    if len(valid) == 0:
        return result

    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)

    result.loc[valid.index] = np.where(
        valid <= q1,
        "Low",
        np.where(valid <= q2, "Medium", "High")
    )

    return result


def classify_instances(df):

    metrics = [
        "aspect_ratio",
        "hull_point_ratio",
        "spatial_entropy",
        "cluster_count",
        "clustered_point_ratio",
        "noise_ratio",
        "mst_max_degree",
        "mst_degree_std",
        "mst_leaf_ratio"
    ]

    for metric in metrics:

        if metric in df.columns:

            df[f"{metric}_class"] = classify_metric(
                df[metric]
            )

    return df


# ============================================================
# 10. HEATMAP DAS CARACTERÍSTICAS
# ============================================================

def plot_characteristics_heatmap(df):

    metrics = [
        "aspect_ratio",
        "hull_point_ratio",
        "spatial_entropy",
        "cluster_count",
        "clustered_point_ratio",
        "noise_ratio",
        "mst_max_degree",
        "mst_degree_std",
        "mst_leaf_ratio"
    ]

    metrics = [
        m for m in metrics
        if m in df.columns
    ]

    data = df.set_index("instance")[metrics].copy()

    # Normalização Min-Max por característica
    normalized = (data - data.min()) / (data.max() - data.min())

    # Evita problemas em métricas constantes
    normalized = normalized.fillna(0)

    fig, ax = plt.subplots(
        figsize=(13, max(6, len(df) * 0.5))
    )

    im = ax.imshow(
        normalized,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["instance"])

    ax.set_title("Características estruturais das instâncias TSP")

    fig.colorbar(im, ax=ax, label="Valor normalizado")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "structural_characteristics_heatmap.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"Heatmap salvo em: {output_path}")


# ============================================================
# 11. ASPECT RATIO VS FECHO CONVEXO
# ============================================================

def plot_aspect_vs_hull(df):

    plt.figure(figsize=(9, 6))

    plt.scatter(
        df["aspect_ratio"],
        df["hull_point_ratio"]
    )

    for _, row in df.iterrows():

        plt.annotate(
            row["instance"],
            (row["aspect_ratio"], row["hull_point_ratio"]),
            fontsize=8
        )

    plt.xlabel("Aspect Ratio")
    plt.ylabel("Proporção de pontos no fecho convexo")

    plt.title("Aspect Ratio vs Hull Ratio")

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "aspect_ratio_vs_hull_ratio.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"Gráfico salvo em: {output_path}")


# ============================================================
# 12. EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    tsp_files = list(
        INSTANCES_DIR.rglob("*.tsp")
    )

    print(f"Instâncias encontradas: {len(tsp_files)}")

    if not tsp_files:

        print(f"Nenhum .tsp encontrado em {INSTANCES_DIR}")

        return

    results = []

    for filepath in tsp_files:

        print(f"Analisando {filepath.name}...")

        try:

            result = analyze_instance(filepath)

            results.append(result)

        except Exception as e:

            print(f"Erro ao analisar {filepath.name}: {e}")

    if not results:

        print("Nenhuma instância analisada com sucesso.")

        return

    df = pd.DataFrame(results)

    # Classificação
    df = classify_instances(df)

    # Ordena pelo nome
    df = df.sort_values("instance").reset_index(drop=True)

    # Exporta CSV
    csv_path = OUTPUT_DIR / "instance_characteristics.csv"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("RESULTADOS DA CLASSIFICAÇÃO")
    print("=" * 70)

    print(df.to_string(index=False))

    print(f"\nCSV salvo em: {csv_path}")

    # Gráficos
    plot_characteristics_heatmap(df)

    plot_aspect_vs_hull(df)


if __name__ == "__main__":
    main()