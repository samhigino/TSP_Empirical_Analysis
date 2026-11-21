from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import networkx as nx
import numpy as np


# Representa uma instância simétrica do TSP carregada a partir de um arquivo TSPLIB.
@dataclass
class TSPGraph:
    name: str
    dimension: int
    edge_weight_type: str
    edge_weight_format: str | None
    dist_matrix: np.ndarray
    coords: np.ndarray | None
    nx_graph: nx.Graph


# Arredonda um valor real para o inteiro mais próximo conforme a função nint da TSPLIB.
def _nint(value: float) -> int:
    return int(value + 0.5) if value >= 0.0 else int(value - 0.5)


# Converte uma coordenada geográfica no formato DDD.MM da TSPLIB para radianos.
def _geo_to_radians(coord: float) -> float:
    pi = 3.141592
    # A documentação/FAQ da TSPLIB especifica truncamento para zero na parte inteira (graus).
    deg = int(coord)
    minutes = coord - deg
    return pi * (deg + 5.0 * minutes / 3.0) / 180.0


# Calcula a matriz de distâncias inteiras simétricas a partir de coordenadas 2D (EUC_2D ou GEO).
def _build_matrix_from_coords(coords: np.ndarray, weight_type: str) -> np.ndarray:
    n = coords.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.int64)

    if weight_type == "EUC_2D":
        for i in range(n):
            for j in range(i + 1, n):
                xd = coords[i, 0] - coords[j, 0]
                yd = coords[i, 1] - coords[j, 1]
                d = _nint(math.sqrt(xd * xd + yd * yd))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    elif weight_type == "GEO":
        rrr = 6378.388
        lat = np.array([_geo_to_radians(coords[i, 0]) for i in range(n)], dtype=np.float64)
        lon = np.array([_geo_to_radians(coords[i, 1]) for i in range(n)], dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                q1 = math.cos(lon[i] - lon[j])
                q2 = math.cos(lat[i] - lat[j])
                q3 = math.cos(lat[i] + lat[j])
                # Clamping numérico evita erro de domínio em acos por imprecisão de ponto flutuante.
                cos_arg = max(-1.0, min(1.0, 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)))
                d = int(rrr * math.acos(cos_arg) + 1.0)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    else:
        raise ValueError(f"Tipo de peso por coordenadas não suportado: {weight_type}")

    return dist_matrix


# Reconstrói a matriz de distâncias simétricas a partir dos tokens explícitos e do formato TSPLIB.
def _build_matrix_from_explicit(tokens: list[float], dim: int, weight_format: str) -> np.ndarray:
    dist_matrix = np.zeros((dim, dim), dtype=np.int64)
    idx = 0

    if weight_format == "FULL_MATRIX":
        for i in range(dim):
            for j in range(dim):
                dist_matrix[i, j] = _nint(tokens[idx])
                idx += 1

    elif weight_format == "UPPER_ROW":
        for i in range(dim - 1):
            for j in range(i + 1, dim):
                val = _nint(tokens[idx])
                dist_matrix[i, j] = val
                dist_matrix[j, i] = val
                idx += 1

    elif weight_format == "LOWER_ROW":
        for i in range(1, dim):
            for j in range(i):
                val = _nint(tokens[idx])
                dist_matrix[i, j] = val
                dist_matrix[j, i] = val
                idx += 1

    elif weight_format == "UPPER_DIAG_ROW":
        for i in range(dim):
            for j in range(i, dim):
                val = _nint(tokens[idx])
                dist_matrix[i, j] = val
                dist_matrix[j, i] = val
                idx += 1

    elif weight_format == "LOWER_DIAG_ROW":
        for i in range(dim):
            for j in range(i + 1):
                val = _nint(tokens[idx])
                dist_matrix[i, j] = val
                dist_matrix[j, i] = val
                idx += 1

    elif weight_format in {"UPPER_COL", "LOWER_COL", "UPPER_DIAG_COL", "LOWER_DIAG_COL"}:
        # Por simetria do TSP, os formatos por coluna correspondem aos respectivos formatos por linha.
        row_equiv = weight_format.replace("_COL", "_ROW")
        return _build_matrix_from_explicit(tokens, dim, row_equiv)

    else:
        raise ValueError(f"Formato de matriz explícita não suportado: {weight_format}")

    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix


# Resolve o caminho do arquivo .tsp a partir de um caminho direto ou do nome da instância na pasta data/.
def _resolve_tsp_path(file_or_name: str | Path) -> Path:
    candidate = Path(file_or_name)
    if candidate.is_file():
        return candidate

    filename = candidate.name if candidate.suffix == ".tsp" else f"{candidate.name}.tsp"
    project_root = Path(__file__).resolve().parent.parent
    data_candidate = project_root / "data" / filename
    if data_candidate.is_file():
        return data_candidate

    raise FileNotFoundError(f"Arquivo de instância TSPLIB não encontrado: {file_or_name}")


# Lê e interpreta um arquivo de instância da TSPLIB (EUC_2D, GEO ou MATRIX) retornando um TSPGraph.
def load_tsp_instance(file_or_name: str | Path) -> TSPGraph:
    filepath = _resolve_tsp_path(file_or_name)
    headers: dict[str, str] = {}
    node_coord_tokens: list[float] = []
    edge_weight_tokens: list[float] = []
    display_coord_tokens: list[float] = []
    current_section: str | None = None

    with filepath.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == "EOF":
                break

            # Identifica início de seções de dados da TSPLIB.
            if line.startswith("NODE_COORD_SECTION"):
                current_section = "NODE_COORD_SECTION"
                continue
            if line.startswith("EDGE_WEIGHT_SECTION"):
                current_section = "EDGE_WEIGHT_SECTION"
                continue
            if line.startswith("DISPLAY_DATA_SECTION"):
                current_section = "DISPLAY_DATA_SECTION"
                continue
            if line.startswith(("FIXED_EDGES_SECTION", "TOUR_SECTION", "DEMAND_SECTION", "DEPOT_SECTION")):
                current_section = "OTHER"
                continue

            if current_section is None:
                if ":" in line:
                    key, val = line.split(":", 1)
                    headers[key.strip().upper()] = val.strip()
            elif current_section == "NODE_COORD_SECTION":
                node_coord_tokens.extend(float(tok) for tok in line.split())
            elif current_section == "EDGE_WEIGHT_SECTION":
                edge_weight_tokens.extend(float(tok) for tok in line.split())
            elif current_section == "DISPLAY_DATA_SECTION":
                display_coord_tokens.extend(float(tok) for tok in line.split())

    name = headers.get("NAME", filepath.stem)
    dimension = int(headers["DIMENSION"])
    weight_type = headers["EDGE_WEIGHT_TYPE"].upper()
    weight_format = headers.get("EDGE_WEIGHT_FORMAT", "").upper() or None

    coords: np.ndarray | None = None
    if node_coord_tokens:
        # Cada registro em NODE_COORD_SECTION possui <id> <x> <y>.
        raw_coords = np.array(node_coord_tokens, dtype=np.float64).reshape(dimension, 3)
        coords = raw_coords[:, 1:3]
    elif display_coord_tokens:
        # Coordenadas auxiliares para visualização (úteis para o fecho convexo em instâncias MATRIX).
        raw_disp = np.array(display_coord_tokens, dtype=np.float64).reshape(dimension, 3)
        coords = raw_disp[:, 1:3]

    if weight_type in {"EUC_2D", "GEO"}:
        if coords is None:
            raise ValueError(f"Instância {name} ({weight_type}) sem NODE_COORD_SECTION.")
        dist_matrix = _build_matrix_from_coords(coords, weight_type)
    elif weight_type == "EXPLICIT":
        if not weight_format:
            raise ValueError(f"Instância {name} (EXPLICIT) sem EDGE_WEIGHT_FORMAT.")
        dist_matrix = _build_matrix_from_explicit(edge_weight_tokens, dimension, weight_format)
    else:
        raise ValueError(f"EDGE_WEIGHT_TYPE não suportado: {weight_type}")

    # Constrói o grafo completo não-direcionado no NetworkX para suporte às rotinas de grafos.
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(dimension))
    for i in range(dimension):
        for j in range(i + 1, dimension):
            nx_graph.add_edge(i, j, weight=int(dist_matrix[i, j]))

    return TSPGraph(
        name=name,
        dimension=dimension,
        edge_weight_type=weight_type,
        edge_weight_format=weight_format,
        dist_matrix=dist_matrix,
        coords=coords,
        nx_graph=nx_graph,
    )