# ruff: noqa: E501
"""Reproducible semantic atlas and distinct-technology selection.

The atlas treats simulator records as evidence and generated pictures as explanatory
illustrations.  Technology ranking is based on recorded lifetime peak performance;
semantic embeddings are used only to prevent near-duplicate featured designs and to
lay out the exploratory map.  No image-derived quantity enters the ranking.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import textwrap
import time
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .technology_visualization import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_STYLES,
    DISCLAIMER,
    IMAGE_STYLES,
    _generate_image,
    _slug,
    build_technology_image_prompt,
    extract_technologies_from_trace,
    render_technology_briefing_sheet,
    technology_controller_signature,
)

ATLAS_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_LOCAL_EMBEDDING_MODEL = "google/embeddinggemma-300m"
CONDITION_ORDER = (
    "full",
    "no-explicit-culture",
    "no-communication",
    "independent-search",
)
CONDITION_LABELS = {
    "full": "Full",
    "no-explicit-culture": "No explicit culture",
    "no-communication": "No communication",
    "independent-search": "Independent search",
}
CONDITION_COLORS = {
    "full": "#177E89",
    "no-explicit-culture": "#E09F3E",
    "no-communication": "#9C6ADE",
    "independent-search": "#65737E",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _member_index(path: Path) -> int | None:
    match = re.search(r"-member-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _technology_uid(
    condition: str,
    population_size: int,
    seed: int,
    trace_path: Path,
    artifact_id: str,
) -> str:
    member = _member_index(trace_path)
    member_part = f"-m{member:03d}" if member is not None else ""
    return f"{condition}-n{population_size}-s{seed}{member_part}-{artifact_id}"


def _ecology_lookup(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    ecology = row.get("technology_ecology", {})
    artifacts = ecology.get("artifacts", []) if isinstance(ecology, Mapping) else []
    return {
        str(item.get("artifact_id")): dict(item)
        for item in artifacts
        if isinstance(item, Mapping) and item.get("artifact_id")
    }


def collect_study_technologies(
    summary_paths: Sequence[str | Path],
    *,
    conditions: Iterable[str] = CONDITION_ORDER,
    include_independent_members: bool = True,
    include_retired: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect uniquely identified artifact records from canonical study summaries."""

    allowed = set(conditions)
    cells: dict[tuple[str, int, int], tuple[Path, dict[str, Any]]] = {}
    duplicates: list[dict[str, Any]] = []
    for raw_summary in summary_paths:
        summary = Path(raw_summary).resolve()
        records = json.loads(summary.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"study summary must contain a JSON list: {summary}")
        for row_value in records:
            row = dict(row_value)
            condition = str(row.get("condition", ""))
            if condition not in allowed:
                continue
            population = int(row.get("population_size", row.get("agents", 0)))
            seed = int(row.get("seed", 0))
            key = (condition, population, seed)
            if key in cells:
                previous = cells[key][0]
                duplicates.append(
                    {"cell": list(key), "kept": str(previous), "ignored": str(summary)}
                )
                continue
            cells[key] = (summary, row)

    trace_jobs: list[tuple[Path, dict[str, Any], Path, bool]] = []
    for (condition, _population, _seed), (summary, row) in sorted(
        cells.items(),
        key=lambda item: (
            CONDITION_ORDER.index(item[0][0])
            if item[0][0] in CONDITION_ORDER
            else len(CONDITION_ORDER),
            item[0][1],
            item[0][2],
        ),
    ):
        selected_output = Path(str(row.get("output", ""))).resolve()
        outputs: list[Path]
        if condition == "independent-search" and include_independent_members:
            outputs = [Path(str(value)).resolve() for value in row.get("member_outputs", [])]
        else:
            outputs = [selected_output]
        for trace in outputs:
            trace_jobs.append((summary, row, trace, trace == selected_output))

    technologies: list[dict[str, Any]] = []
    missing: list[str] = []
    progress = tqdm(trace_jobs, desc="Collecting technology traces", unit="trace")
    for summary, row, trace, is_selected_output in progress:
        progress.set_postfix_str(trace.name[-42:])
        if not trace.is_file():
            missing.append(str(trace))
            continue
        extracted = extract_technologies_from_trace(
            trace,
            include_retired=include_retired,
            show_progress=False,
        )
        condition = str(row["condition"])
        population = int(row.get("population_size", row.get("agents", 0)))
        seed = int(row["seed"])
        ecology = _ecology_lookup(row) if is_selected_output else {}
        for technology in extracted:
            artifact_id = str(technology.get("artifact_id", ""))
            enriched = dict(technology)
            if artifact_id in ecology:
                for key, value in ecology[artifact_id].items():
                    enriched.setdefault(key, _jsonable(value))
            enriched.update(
                {
                    "technology_uid": _technology_uid(
                        condition, population, seed, trace, artifact_id
                    ),
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(condition, condition),
                    "population_size": population,
                    "seed": seed,
                    "independent_member": _member_index(trace),
                    "source_summary": str(summary),
                }
            )
            technologies.append(enriched)

    technologies.sort(key=lambda item: str(item["technology_uid"]))
    uids = [str(item["technology_uid"]) for item in technologies]
    if len(set(uids)) != len(uids):
        counts = Counter(uids)
        repeated = sorted(uid for uid, count in counts.items() if count > 1)
        raise ValueError(f"duplicate technology UIDs: {repeated[:5]}")
    audit = {
        "atlas_version": ATLAS_VERSION,
        "study_summaries": [str(Path(path).resolve()) for path in summary_paths],
        "conditions": sorted(allowed),
        "study_cell_count": len(cells),
        "trace_count": len(trace_jobs),
        "technology_count": len(technologies),
        "technology_count_by_condition": dict(
            sorted(Counter(str(item["condition"]) for item in technologies).items())
        ),
        "duplicate_cells_ignored": duplicates,
        "missing_traces": missing,
        "include_independent_members": include_independent_members,
    }
    return technologies, audit


def _collection_cache_key(
    summary_paths: Sequence[str | Path],
    *,
    conditions: Iterable[str],
    include_independent_members: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "atlas_version": ATLAS_VERSION,
                "conditions": sorted(set(conditions)),
                "include_independent_members": include_independent_members,
            },
            sort_keys=True,
        ).encode()
    )
    for raw_path in summary_paths:
        path = Path(raw_path).resolve()
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def collect_study_technologies_cached(
    summary_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    conditions: Iterable[str] = CONDITION_ORDER,
    include_independent_members: bool = True,
    include_retired: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cache the expensive multi-trace collection without weakening provenance."""

    destination = Path(output_dir).resolve()
    catalog_path = destination / "technology-source-catalog.json"
    audit_path = destination / "technology-collection-audit.json"
    selected_conditions = tuple(conditions)
    cache_key = _collection_cache_key(
        summary_paths,
        conditions=selected_conditions,
        include_independent_members=include_independent_members,
    )
    if catalog_path.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("collection_cache_key") == cache_key:
            technologies = json.loads(catalog_path.read_text(encoding="utf-8"))
            audit["loaded_from_cache"] = True
            return technologies, audit
    technologies, audit = collect_study_technologies(
        summary_paths,
        conditions=selected_conditions,
        include_independent_members=include_independent_members,
        include_retired=include_retired,
    )
    audit["collection_cache_key"] = cache_key
    audit["loaded_from_cache"] = False
    _write_json(catalog_path, technologies)
    _write_json(audit_path, audit)
    return technologies, audit


def technology_semantic_text(technology: Mapping[str, Any]) -> str:
    """Return the recorded, deterministic text representation used for embeddings."""

    recipe = technology.get("recipe", {})
    if not isinstance(recipe, Mapping):
        recipe = {}
    program = technology.get("initial_program", {})
    if not isinstance(program, Mapping):
        program = {}
    composition = technology.get("composition", {})
    composition_names = list(composition) if isinstance(composition, Mapping) else []
    inputs = recipe.get("inputs", [])
    input_names = [
        str(item.get("resource", ""))
        for item in inputs
        if isinstance(item, Mapping) and item.get("resource")
    ]
    steps = recipe.get("steps", [])
    process_names = [
        str(item.get("operation", ""))
        for item in steps
        if isinstance(item, Mapping) and item.get("operation")
    ]
    instructions = program.get("instructions", [])
    controller_operations = [
        str(item.get("op", ""))
        for item in instructions
        if isinstance(item, Mapping) and item.get("op")
    ]
    parts = [
        str(technology.get("name", "")),
        str(technology.get("architecture", "")),
        str(technology.get("claimed_function", "")),
        ", ".join(map(str, technology.get("bio_inspiration", []))),
        ", ".join(map(str, technology.get("predicted_effects", []))),
        "material constituents " + ", ".join(map(str, composition_names + input_names)),
        "fabrication sequence " + " then ".join(process_names),
        "fabricated form " + str(recipe.get("output_form", "")),
        "design principles " + ", ".join(map(str, recipe.get("design_principles", []))),
        "adaptive control operations " + ", ".join(controller_operations),
    ]
    return "\n".join(part for part in parts if part.strip())


def _embedding_cache_key(
    technologies: Sequence[Mapping[str, Any]],
    *,
    provider: str,
    model: str,
    dimensions: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{provider}\0{model}\0{dimensions}\0".encode())
    for item in technologies:
        digest.update(str(item["technology_uid"]).encode())
        digest.update(b"\0")
        digest.update(technology_semantic_text(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def build_semantic_embeddings(
    technologies: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    provider: str = "embeddinggemma",
    model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
    dimensions: int = 768,
    batch_size: int = 16,
    overwrite: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed every design, caching the exact ordered matrix and its provenance."""

    if provider not in {"embeddinggemma", "openai", "tfidf"}:
        raise ValueError("embedding provider must be 'embeddinggemma', 'openai', or 'tfidf'")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "technology-embeddings.npz"
    metadata_path = destination / "technology-embeddings.json"
    cache_key = _embedding_cache_key(
        technologies, provider=provider, model=model, dimensions=dimensions
    )
    if not overwrite and matrix_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") == cache_key:
            matrix = np.load(matrix_path)["embeddings"]
            return np.asarray(matrix, dtype=np.float32), metadata

    texts = [technology_semantic_text(item) for item in technologies]
    device = None
    if provider == "embeddinggemma":
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "EmbeddingGemma requires torch, transformers, and sentence-transformers"
            ) from exc
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        encoder = SentenceTransformer(
            model,
            device=device,
            local_files_only=True,
            model_kwargs={"torch_dtype": torch.float32},
        )
        matrix = np.asarray(
            encoder.encode_document(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if dimensions > matrix.shape[1]:
            raise ValueError(
                f"{model} produces {matrix.shape[1]} dimensions; requested {dimensions}"
            )
        matrix = matrix[:, :dimensions]
    elif provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI embeddings require the openai package") from exc
        client = OpenAI(max_retries=5, timeout=180.0)
        rows: list[list[float]] = []
        starts = range(0, len(texts), batch_size)
        for start in tqdm(
            starts,
            total=math.ceil(len(texts) / batch_size),
            desc="Embedding technologies",
            unit="batch",
        ):
            response = client.embeddings.create(
                model=model,
                input=texts[start : start + batch_size],
                dimensions=dimensions,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            rows.extend(item.embedding for item in ordered)
        matrix = np.asarray(rows, dtype=np.float32)
    else:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12_000,
            sublinear_tf=True,
        )
        sparse = vectorizer.fit_transform(
            tqdm(texts, desc="Vectorizing technologies", unit="technology")
        )
        components = min(dimensions, sparse.shape[0] - 1, sparse.shape[1] - 1)
        if components < 2:
            raise ValueError("at least three technologies are required for TF-IDF embeddings")
        matrix = TruncatedSVD(
            n_components=components,
            algorithm="arpack",
            random_state=42,
        ).fit_transform(sparse)
        model = f"tfidf-svd-{components}"

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, np.maximum(norms, 1e-12)).astype(np.float32)
    metadata = {
        "cache_key": cache_key,
        "provider": provider,
        "model": model,
        "dimensions": int(matrix.shape[1]),
        "device": device,
        "technology_count": len(technologies),
        "ordered_uids": [str(item["technology_uid"]) for item in technologies],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    np.savez_compressed(matrix_path, embeddings=matrix)
    _write_json(metadata_path, metadata)
    return matrix, metadata


def _performance_ranks(technologies: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    performance = np.asarray(
        [max(0.0, float(item.get("rank_performance", 0.0))) for item in technologies],
        dtype=np.float64,
    )
    order = np.lexsort(
        (
            np.asarray([str(item["technology_uid"]) for item in technologies]),
            -performance,
        )
    )
    ranks = np.empty(len(technologies), dtype=np.int64)
    ranks[order] = np.arange(1, len(technologies) + 1)
    return performance, ranks


def select_distinct_top_technologies(
    technologies: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    count: int = 16,
    maximum_cosine_similarity: float = 0.82,
    cluster_assignments: np.ndarray | None = None,
    maximum_per_cluster: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Select the highest-performing designs subject to semantic distinctness.

    Candidates are visited in descending recorded lifetime peak performance.  A design
    is accepted only when it is not an exact text duplicate and its cosine similarity
    to every already selected design is no greater than the declared threshold.
    """

    if count <= 0:
        raise ValueError("featured count must be positive")
    if len(technologies) != len(embeddings):
        raise ValueError("technology and embedding row counts do not match")
    if count > len(technologies):
        raise ValueError("featured count exceeds the technology population")
    if not 0.0 <= maximum_cosine_similarity < 1.0:
        raise ValueError("maximum cosine similarity must be in [0, 1)")
    if maximum_per_cluster is not None and maximum_per_cluster <= 0:
        raise ValueError("maximum_per_cluster must be positive")
    if cluster_assignments is not None and len(cluster_assignments) != len(technologies):
        raise ValueError("cluster assignment row count does not match technologies")

    performance, ranks = _performance_ranks(technologies)
    order = sorted(
        range(len(technologies)),
        key=lambda index: (
            -performance[index],
            str(technologies[index]["technology_uid"]),
        ),
    )
    selected: list[int] = []
    fingerprints: set[str] = set()
    selected_clusters: Counter[int] = Counter()
    nearest_similarities: list[float] = []
    for index in order:
        cluster = int(cluster_assignments[index]) if cluster_assignments is not None else None
        if (
            cluster is not None
            and maximum_per_cluster is not None
            and selected_clusters[cluster] >= maximum_per_cluster
        ):
            continue
        text = " ".join(technology_semantic_text(technologies[index]).lower().split())
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        if fingerprint in fingerprints:
            continue
        maximum = float(np.max(embeddings[selected] @ embeddings[index])) if selected else 0.0
        if selected and maximum > maximum_cosine_similarity:
            continue
        selected.append(index)
        fingerprints.add(fingerprint)
        if cluster is not None:
            selected_clusters[cluster] += 1
        nearest_similarities.append(maximum)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} technologies satisfied maximum cosine similarity "
            f"{maximum_cosine_similarity:.3f}; increase the threshold explicitly"
        )
    audit = {
        "selection_rule": "performance-ranked greedy semantic exclusion",
        "performance_measure": "recorded lifetime peak artifact performance",
        "featured_count": count,
        "maximum_allowed_cosine_similarity": maximum_cosine_similarity,
        "maximum_observed_pair_similarity": max(nearest_similarities, default=0.0),
        "minimum_observed_pair_distance": 1.0 - max(nearest_similarities, default=0.0),
        "maximum_per_semantic_cluster": maximum_per_cluster,
        "selected_cluster_counts": {
            str(key): value for key, value in sorted(selected_clusters.items())
        },
        "selected_global_performance_ranks": [int(ranks[index]) for index in selected],
    }
    return selected, audit


def build_umap_embedding(
    embeddings: np.ndarray,
    *,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.12,
    components: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a deterministic UMAP layout from normalized semantic embeddings."""

    os.environ.setdefault(
        "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "swarmworld-numba-cache")
    )
    try:
        import umap
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "technology-atlas requires umap-learn; install the 'analysis' extra and set "
            "NUMBA_CACHE_DIR to a writable directory"
        ) from exc
    reducer = umap.UMAP(
        n_components=components,
        n_neighbors=min(n_neighbors, max(2, len(embeddings) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        transform_seed=random_state,
        low_memory=True,
        n_jobs=1,
    )
    coordinates = reducer.fit_transform(embeddings).astype(np.float32)
    metadata = {
        "method": "UMAP",
        "package_version": getattr(umap, "__version__", "unknown"),
        "metric": "cosine",
        "components": components,
        "n_neighbors": min(n_neighbors, max(2, len(embeddings) - 1)),
        "min_dist": min_dist,
        "random_state": random_state,
        "n_jobs": 1,
    }
    return coordinates, metadata


def _cluster_labels(
    technologies: Sequence[Mapping[str, Any]],
    assignments: np.ndarray,
    *,
    terms_per_cluster: int = 3,
) -> dict[int, str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    texts = [technology_semantic_text(item) for item in technologies]
    schema_words = {
        "adaptive",
        "const",
        "control",
        "design",
        "fabricated",
        "fabrication",
        "form",
        "operations",
        "principles",
        "sequence",
    }
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(ENGLISH_STOP_WORDS | schema_words),
        ngram_range=(1, 2),
        max_features=8000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )
    sparse = vectorizer.fit_transform(texts)
    terms = np.asarray(vectorizer.get_feature_names_out())
    labels: dict[int, str] = {}
    for cluster in sorted(set(map(int, assignments))):
        rows = np.flatnonzero(assignments == cluster)
        weights = np.asarray(sparse[rows].mean(axis=0)).ravel()
        best = terms[np.argsort(weights)[::-1][:terms_per_cluster]]
        labels[cluster] = " · ".join(map(str, best))
    return labels


def cluster_technologies(
    technologies: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    random_state: int = 42,
    minimum_clusters: int = 3,
    maximum_clusters: int = 12,
) -> tuple[np.ndarray, dict[int, str], dict[str, Any]]:
    """Choose K by silhouette score in the original semantic embedding space."""

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    features = np.asarray(embeddings, dtype=np.float64)
    scores: dict[int, float] = {}
    models: dict[int, Any] = {}
    upper = min(maximum_clusters, len(technologies) - 1)
    for cluster_count in tqdm(
        range(minimum_clusters, upper + 1),
        desc="Selecting semantic clusters",
        unit="K",
    ):
        model = AgglomerativeClustering(
            n_clusters=cluster_count,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(features)
        # Apple Accelerate can emit a spurious ``matmul`` RuntimeWarning here even
        # when every input and resulting score is finite. Suppress that narrow
        # warning, then explicitly reject any genuinely non-finite score.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*encountered in matmul", category=RuntimeWarning
            )
            score = silhouette_score(
                features,
                labels,
                metric="cosine",
                sample_size=min(2000, len(features)),
                random_state=random_state,
            )
        if not math.isfinite(float(score)):
            raise RuntimeError(f"non-finite silhouette score for {cluster_count} semantic clusters")
        scores[cluster_count] = float(score)
        models[cluster_count] = model
    best_k = max(scores, key=lambda value: (scores[value], -value))
    assignments = np.asarray(models[best_k].labels_, dtype=np.int64)
    labels = _cluster_labels(technologies, assignments)
    metadata = {
        "method": "average-linkage agglomerative clustering of semantic embeddings",
        "selection": "maximum silhouette score",
        "random_state": random_state,
        "cluster_count": best_k,
        "candidate_silhouette_scores": {str(key): value for key, value in scores.items()},
        "labels": {str(key): value for key, value in labels.items()},
    }
    return assignments, labels, metadata


def _set_paper_style() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "biofoundry-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_figure(figure: Any, output_dir: Path, stem: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(
            path,
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs[suffix] = str(path)
    return outputs


def render_technology_atlas_figures(
    technologies: Sequence[Mapping[str, Any]],
    coordinates: np.ndarray,
    selected_indices: Sequence[int],
    output_dir: str | Path,
) -> dict[str, dict[str, str]]:
    """Render the deterministic two-dimensional paper figures."""

    _set_paper_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected_lookup = {index: rank for rank, index in enumerate(selected_indices, start=1)}
    conditions = [str(item["condition"]) for item in technologies]

    figure, axis = plt.subplots(figsize=(9.4, 7.4), constrained_layout=True)
    for condition in CONDITION_ORDER:
        rows = np.asarray([value == condition for value in conditions])
        if not np.any(rows):
            continue
        axis.scatter(
            coordinates[rows, 0],
            coordinates[rows, 1],
            s=9,
            alpha=0.48,
            linewidths=0,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
            rasterized=False,
        )
    for index, rank in selected_lookup.items():
        axis.scatter(
            coordinates[index, 0],
            coordinates[index, 1],
            s=72,
            color=CONDITION_COLORS.get(conditions[index], "#333333"),
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        axis.annotate(
            str(rank),
            (coordinates[index, 0], coordinates[index, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="#17252A",
            zorder=6,
        )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("Semantic landscape of trace-derived technologies", loc="left")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)
    landscape = _save_figure(figure, destination, "technology-semantic-landscape")
    plt.close(figure)

    performance = [
        float(technologies[index].get("rank_performance", 0.0)) for index in selected_indices
    ]
    labels = [
        f"{rank:02d}  {str(technologies[index].get('name', 'technology'))[:48]}"
        for rank, index in enumerate(selected_indices, start=1)
    ]
    ranking, axis = plt.subplots(figsize=(9.4, 6.8), constrained_layout=True)
    positions = np.arange(len(selected_indices))[::-1]
    bars = axis.barh(
        positions,
        performance,
        color=[CONDITION_COLORS.get(conditions[index], "#65737E") for index in selected_indices],
        alpha=0.9,
    )
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Recorded lifetime peak performance")
    axis.set_title("Sixteen performance-ranked, semantically distinct technologies", loc="left")
    for bar, value in zip(bars, performance, strict=True):
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=7.5,
        )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor=CONDITION_COLORS[condition],
                markeredgewidth=0,
                label=CONDITION_LABELS[condition],
            )
            for condition in CONDITION_ORDER
            if condition in conditions
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
    )
    ranked = _save_figure(ranking, destination, "featured-technology-ranking")
    plt.close(ranking)
    return {"semantic_landscape": landscape, "featured_ranking": ranked}


def _plain_explanation(technology: Mapping[str, Any]) -> str:
    architecture = str(technology.get("architecture") or "an unrecorded architecture")
    function = str(technology.get("claimed_function") or "an unrecorded intended function")
    bio = ", ".join(map(str, technology.get("bio_inspiration", [])))
    inspiration = f" It draws on {bio}." if bio else ""
    return f"The agent designed {architecture} Its intended role was {function}.{inspiration}"


def _catalog_record(
    technology: Mapping[str, Any],
    coordinates: np.ndarray,
    index: int,
    *,
    cluster: int,
    cluster_label: str,
    performance_rank: int,
    feature_rank: int | None,
    dossier_dir: str | None,
) -> dict[str, Any]:
    services = technology.get("lifetime_peak_services") or technology.get("services") or {}
    controller = technology_controller_signature(technology)
    return {
        "technology_uid": technology["technology_uid"],
        "artifact_id": technology.get("artifact_id"),
        "name": technology.get("name"),
        "condition": technology.get("condition"),
        "condition_label": technology.get("condition_label"),
        "population_size": technology.get("population_size"),
        "seed": technology.get("seed"),
        "independent_member": technology.get("independent_member"),
        "creator": technology.get("creator"),
        "created_tick": technology.get("created_tick"),
        "performance": float(technology.get("rank_performance", 0.0)),
        "performance_rank": performance_rank,
        "featured_rank": feature_rank,
        "cluster": cluster,
        "cluster_label": cluster_label,
        "umap": [float(value) for value in coordinates[index]],
        "architecture": technology.get("architecture"),
        "claimed_function": technology.get("claimed_function"),
        "explanation": _plain_explanation(technology),
        "bio_inspiration": technology.get("bio_inspiration", []),
        "predicted_effects": technology.get("predicted_effects", []),
        "geometry": technology.get("geometry", {}),
        "composition": technology.get("composition", {}),
        "recipe": technology.get("recipe", {}),
        "controller": controller,
        "services": services,
        "contributors": technology.get("contributors", []),
        "participants": technology.get("participants", []),
        "collaborative": technology.get("collaborative", False),
        "citation_depth": technology.get("citation_depth", 0),
        "source_trace": technology.get("source_trace"),
        "source_trace_sha256": technology.get("source_trace_sha256"),
        "dossier_dir": dossier_dir,
        "engineering_image": f"{dossier_dir}/engineering.png" if dossier_dir else None,
        "photorealistic_image": f"{dossier_dir}/photorealistic.png" if dossier_dir else None,
        "overview_image": f"{dossier_dir}/overview.png" if dossier_dir else None,
        "mechanism_image": f"{dossier_dir}/mechanism.png" if dossier_dir else None,
    }


def write_threejs_explorer(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Write a Three.js explorer driven by a portable JSON data product."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "technology-explorer-data.json", list(records))
    repository = Path(__file__).resolve().parents[2]
    three_source = repository / "web/node_modules/three/build/three.module.js"
    three_core_source = repository / "web/node_modules/three/build/three.core.js"
    controls_source = repository / "web/node_modules/three/examples/jsm/controls/OrbitControls.js"
    if three_source.is_file() and three_core_source.is_file() and controls_source.is_file():
        vendor = destination / "vendor"
        controls_vendor = vendor / "controls"
        controls_vendor.mkdir(parents=True, exist_ok=True)
        shutil.copy2(three_source, vendor / "three.module.js")
        shutil.copy2(three_core_source, vendor / "three.core.js")
        shutil.copy2(controls_source, controls_vendor / "OrbitControls.js")
        import_map = {
            "imports": {
                "three": "./vendor/three.module.js",
                "three/addons/": "./vendor/",
            }
        }
    else:
        import_map = {
            "imports": {
                "three": ("https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js"),
                "three/addons/": ("https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"),
            }
        }
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwarmWorld Technology Atlas</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;color:#eaf2ef;background:#071310;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;overflow:hidden}#app{display:grid;grid-template-columns:minmax(0,1fr) 390px;height:100vh}
#stage{position:relative;min-width:0;background:radial-gradient(circle at 50% 42%,#123229 0,#081a15 48%,#050d0b 100%)}
#canvas{position:absolute;inset:0}.toolbar{position:absolute;z-index:5;left:18px;top:16px;right:18px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.brand{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#7bd5b4;margin-right:8px}.control{height:34px;border:1px solid #31554a;background:#0d211b;color:#eaf2ef;border-radius:7px;padding:0 10px}
input.control{width:230px}button.control{cursor:pointer}.legend{position:absolute;left:18px;bottom:15px;z-index:5;display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#c8d8d2}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.hint{position:absolute;right:16px;bottom:14px;color:#74958a;font-size:11px;z-index:5}
#detail{background:#f7faf8;color:#183028;border-left:1px solid #29463d;overflow:auto;padding:22px 22px 44px}#detail h1{font-size:24px;line-height:1.1;margin:6px 0 10px}
.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:10px;color:#347562;font-weight:700}.meta{font-size:12px;color:#5e756d;line-height:1.55}.metric{font-size:27px;font-weight:720;color:#116b54;margin:14px 0 3px}
.section{border-top:1px solid #dce6e1;margin-top:18px;padding-top:15px}.section h2{font-size:11px;text-transform:uppercase;letter-spacing:.11em;color:#527169;margin:0 0 8px}.section p,.section pre{font:13px/1.5 Inter,ui-sans-serif,system-ui;margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
.images{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}.images img{width:100%;aspect-ratio:3/2;object-fit:cover;background:#e6eeea;border:1px solid #d4e0da;border-radius:5px}.tag{display:inline-block;border:1px solid #b9d0c6;border-radius:999px;padding:3px 7px;margin:2px 3px 2px 0;font-size:11px;color:#28594b}
.empty{margin-top:38vh;text-align:center;color:#6c837b}.tooltip{position:absolute;z-index:8;pointer-events:none;background:#f7faf8;color:#183028;border:1px solid #c8d9d2;padding:7px 9px;border-radius:5px;font-size:11px;display:none;max-width:270px;box-shadow:0 8px 28px #0008}
#labels{position:absolute;inset:0;overflow:hidden;pointer-events:none}.label{position:absolute;z-index:4;pointer-events:none;color:#fff;background:#071310d9;border:1px solid #7bd5b488;border-radius:50%;width:23px;height:23px;text-align:center;line-height:21px;font-size:10px;font-weight:700}
@media(max-width:850px){#app{grid-template-columns:1fr}#detail{position:absolute;z-index:9;right:0;width:min(390px,92vw);height:100vh;transform:translateX(100%);transition:transform .2s}#detail.open{transform:none}}
</style>
<script type="importmap">__THREE_IMPORT_MAP__</script></head>
<body><div id="app"><main id="stage"><div id="canvas"></div><div class="toolbar"><span class="brand">Technology Atlas</span><input id="search" class="control" placeholder="Search designs, functions, biology…"><select id="condition" class="control"><option value="all">All conditions</option><option value="full">Full</option><option value="no-explicit-culture">No explicit culture</option><option value="no-communication">No communication</option><option value="independent-search">Independent search</option></select><select id="population" class="control"><option value="all">All population sizes</option><option value="50">N = 50</option><option value="100">N = 100</option><option value="200">N = 200</option></select><button id="featured" class="control">Featured 16</button><button id="reset" class="control">Reset view</button></div><div id="labels"></div><div id="tooltip" class="tooltip"></div><div class="legend"><span><i class="dot" style="background:#177E89"></i>Full</span><span><i class="dot" style="background:#E09F3E"></i>No explicit culture</span><span><i class="dot" style="background:#9C6ADE"></i>No communication</span><span><i class="dot" style="background:#65737E"></i>Independent</span></div><div class="hint">Drag to rotate · scroll to zoom · click a technology</div></main><aside id="detail"><div class="empty">Select a technology to inspect its design and provenance.</div></aside></div>
<script type="module">
import * as THREE from 'three';import{OrbitControls}from'three/addons/controls/OrbitControls.js';
const data=await(await fetch('./technology-explorer-data.json')).json();const host=document.getElementById('canvas'),detail=document.getElementById('detail'),tooltip=document.getElementById('tooltip'),labelsHost=document.getElementById('labels');
const colors={full:0x177e89,'no-explicit-culture':0xe09f3e,'no-communication':0x9c6ade,'independent-search':0x65737e};
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.1,1000);camera.position.set(0,0,32);const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.07;controls.minDistance=6;controls.maxDistance=150;
const coords=data.map(d=>d.umap),extent=a=>Math.max(...a)-Math.min(...a),scale=22/Math.max(extent(coords.map(d=>d[0])),extent(coords.map(d=>d[1])),extent(coords.map(d=>d[2]||0)),1),center=[0,1,2].map(k=>(Math.max(...coords.map(d=>d[k]||0))+Math.min(...coords.map(d=>d[k]||0)))/2);
const geometry=new THREE.SphereGeometry(.12,9,7),material=new THREE.MeshBasicMaterial({transparent:true,opacity:.9,toneMapped:false}),mesh=new THREE.InstancedMesh(geometry,material,data.length);mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);const object=new THREE.Object3D(),color=new THREE.Color();
const positions=data.map(d=>new THREE.Vector3((d.umap[0]-center[0])*scale,(d.umap[1]-center[1])*scale,(d.umap[2]-center[2])*scale));
let visible=data.map(()=>true),featuredOnly=false;function update(){const q=document.getElementById('search').value.toLowerCase(),c=document.getElementById('condition').value,n=document.getElementById('population').value;data.forEach((d,i)=>{const hay=(d.name+' '+d.architecture+' '+d.claimed_function+' '+d.bio_inspiration.join(' ')).toLowerCase();visible[i]=(!q||hay.includes(q))&&(c==='all'||d.condition===c)&&(n==='all'||String(d.population_size)===n)&&(!featuredOnly||d.featured_rank);object.position.copy(positions[i]);object.scale.setScalar(visible[i]?(d.featured_rank?1.65:.65):0);object.updateMatrix();mesh.setMatrixAt(i,object.matrix);color.setHex(colors[d.condition]||0x65737e);mesh.setColorAt(i,color)});mesh.instanceMatrix.needsUpdate=true;mesh.instanceColor.needsUpdate=true;renderLabels()}
scene.add(mesh);const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();function hit(e){const r=renderer.domElement.getBoundingClientRect();mouse.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(mouse,camera);return ray.intersectObject(mesh).find(x=>visible[x.instanceId])}
function text(v){return String(v??'Not recorded').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}function json(v){return text(JSON.stringify(v,null,2))}
function show(d){const imgs=d.dossier_dir?`<div class="images"><img src="${text(d.engineering_image)}" alt="Engineering concept"><img src="${text(d.photorealistic_image)}" alt="Photorealistic concept"></div>`:'';detail.innerHTML=`<div class="eyebrow">${d.featured_rank?'Featured '+String(d.featured_rank).padStart(2,'0'):'Technology '+d.performance_rank}</div><h1>${text(d.name)}</h1><div class="meta">${text(d.condition_label)} · N=${d.population_size} · seed ${d.seed} · tick ${d.created_tick}<br>${text(d.creator)} · ${text(d.technology_uid)}</div><div class="metric">${d.performance.toFixed(3)}</div><div class="meta">recorded lifetime peak performance · global rank ${d.performance_rank}</div>${imgs}<div class="section"><h2>Plain-language explanation</h2><p>${text(d.explanation)}</p></div><div class="section"><h2>Biological inspiration</h2><p>${d.bio_inspiration.map(x=>`<span class="tag">${text(x)}</span>`).join('')||'Not recorded'}</p></div><div class="section"><h2>Architecture</h2><p>${text(d.architecture)}</p></div><div class="section"><h2>Claimed function</h2><p>${text(d.claimed_function)}</p></div><div class="section"><h2>Semantic family</h2><p>${text(d.cluster_label)}</p></div><div class="section"><h2>Composition</h2><pre>${json(d.composition)}</pre></div><div class="section"><h2>Recipe</h2><pre>${json(d.recipe)}</pre></div><div class="section"><h2>Recorded services</h2><pre>${json(d.services)}</pre></div><div class="section"><h2>Provenance</h2><p>${text(d.source_trace)}<br>SHA-256: ${text(d.source_trace_sha256)}</p></div>`;detail.classList.add('open')}
renderer.domElement.addEventListener('pointermove',e=>{const h=hit(e);if(!h){tooltip.style.display='none';return}const d=data[h.instanceId];tooltip.innerHTML=`<b>${text(d.name)}</b><br>${text(d.condition_label)} · ${d.performance.toFixed(3)}`;tooltip.style.display='block';tooltip.style.left=(e.clientX+12)+'px';tooltip.style.top=(e.clientY+12)+'px'});renderer.domElement.addEventListener('click',e=>{const h=hit(e);if(h)show(data[h.instanceId])});
function renderLabels(){labelsHost.innerHTML='';data.forEach((d,i)=>{if(!d.featured_rank||!visible[i])return;const el=document.createElement('div');el.className='label';el.textContent=d.featured_rank;el.dataset.i=i;labelsHost.appendChild(el)})}function positionLabels(){labelsHost.querySelectorAll('.label').forEach(el=>{const p=positions[+el.dataset.i].clone().project(camera),r=host.getBoundingClientRect();el.style.left=((p.x*.5+.5)*r.width-11)+'px';el.style.top=((-p.y*.5+.5)*r.height-11)+'px';el.style.display=p.z<1?'block':'none'})}
['search','condition','population'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',update));document.getElementById('featured').onclick=e=>{featuredOnly=!featuredOnly;e.currentTarget.textContent=featuredOnly?'Show all':'Featured 16';update()};document.getElementById('reset').onclick=()=>{camera.position.set(0,0,32);controls.target.set(0,0,0);controls.update()};
function resize(){const r=host.getBoundingClientRect();camera.aspect=r.width/r.height;camera.updateProjectionMatrix();renderer.setSize(r.width,r.height,false)}new ResizeObserver(resize).observe(host);function animate(){requestAnimationFrame(animate);controls.update();positionLabels();renderer.render(scene,camera)}update();resize();animate();
</script></body></html>"""
    explorer = destination / "technology-explorer.html"
    explorer.write_text(
        html.replace("__THREE_IMPORT_MAP__", json.dumps(import_map, separators=(",", ":"))),
        encoding="utf-8",
    )
    return explorer


def _generate_featured_dossiers(
    technologies: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    output_dir: Path,
    *,
    generate: bool,
    model: str,
    size: str,
    quality: str,
    overwrite: bool,
    workers: int,
    styles: Sequence[str] = DEFAULT_IMAGE_STYLES,
) -> tuple[dict[int, str], dict[str, Any]]:
    featured_root = output_dir / "featured"
    featured_root.mkdir(parents=True, exist_ok=True)
    client = None
    if generate:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("image generation requires the openai package") from exc
        client = OpenAI(max_retries=5, timeout=300.0)
    folders: dict[int, str] = {}
    entries: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    selected_styles = tuple(dict.fromkeys(styles))
    unknown_styles = set(selected_styles) - set(IMAGE_STYLES)
    if unknown_styles:
        raise ValueError(f"unknown image styles: {sorted(unknown_styles)}")
    jobs = [
        (rank, index, style)
        for rank, index in enumerate(selected_indices, 1)
        for style in selected_styles
    ]
    completed: set[int] = set()
    for rank, index, style in jobs:
        technology = technologies[index]
        name = str(technology.get("name") or technology["technology_uid"])
        folder = (
            featured_root / f"{rank:02d}-{_slug(name)}-{_slug(str(technology['technology_uid']))}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        folders[index] = str(folder.relative_to(output_dir))
        if index not in completed:
            _write_json(folder / "technology.json", technology)
            completed.add(index)
        prompt = build_technology_image_prompt(technology, style)
        prompt_path = folder / f"{style}-prompt.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        image_path = folder / f"{style}.png"
        entry = {
            "featured_rank": rank,
            "technology_uid": technology["technology_uid"],
            "style": style,
            "model": model,
            "size": size,
            "quality": quality,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "output": str(image_path.relative_to(output_dir)),
            "status": "prompt_only",
            "error": None,
        }
        if image_path.exists() and not overwrite:
            entry["status"] = "existing"
            entries.append(entry)
        elif generate:
            pending.append({**entry, "image_path": image_path})
        else:
            entries.append(entry)

    def render_one(entry: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(entry.pop("image_path"))
        for attempt in range(1, 4):
            try:
                _generate_image(
                    client=client,
                    model=model,
                    prompt=str(entry["prompt"]),
                    size=size,
                    quality=quality,
                    output=image_path,
                )
                entry["status"] = "generated"
                entry["error"] = None
                return entry
            except Exception as exc:  # continue a resumable, expensive batch
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["status"] = "error"
                if attempt < 3:
                    time.sleep(2**attempt)
        return entry

    if pending:
        worker_count = max(1, min(int(workers), len(pending)))
        with tqdm(
            total=len(jobs),
            initial=len(entries),
            desc="Rendering featured technologies",
            unit="image",
        ) as progress:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(render_one, entry) for entry in pending]
                for future in as_completed(futures):
                    entries.append(future.result())
                    progress.update()
    entries.sort(key=lambda entry: (int(entry["featured_rank"]), str(entry["style"])))

    sheets: list[str] = []
    for index in selected_indices:
        folder = output_dir / folders[index]
        image_paths = {
            style: folder / f"{style}.png"
            for style in IMAGE_STYLES
            if (folder / f"{style}.png").is_file()
        }
        sheet_paths = render_technology_briefing_sheet(
            technologies[index], image_paths, folder / "briefing-sheet"
        )
        sheets.append(sheet_paths["pdf"])
    manifest = {
        "atlas_version": ATLAS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generated": generate,
        "model": model,
        "size": size,
        "quality": quality,
        "workers": max(1, int(workers)),
        "styles": list(selected_styles),
        "entries": entries,
        "briefing_sheets": sheets,
        "disclaimer": DISCLAIMER,
    }
    _write_json(output_dir / "featured-render-manifest.json", manifest)
    return folders, manifest


def _write_catalog_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "technology_uid",
        "artifact_id",
        "name",
        "condition",
        "population_size",
        "seed",
        "independent_member",
        "creator",
        "created_tick",
        "performance",
        "performance_rank",
        "featured_rank",
        "cluster",
        "cluster_label",
        "umap_1",
        "umap_2",
        "umap_3",
        "claimed_function",
        "architecture",
        "source_trace",
        "source_trace_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in records:
            coordinates = list(item["umap"])
            writer.writerow(
                {
                    **{field: item.get(field) for field in fields if not field.startswith("umap_")},
                    "umap_1": coordinates[0],
                    "umap_2": coordinates[1],
                    "umap_3": coordinates[2] if len(coordinates) > 2 else 0.0,
                }
            )


def build_technology_atlas_report(
    output_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    selection_audit: Mapping[str, Any],
) -> Path:
    """Compose the overview figures and featured sheets into one PDF atlas."""

    _set_paper_style()
    import matplotlib.pyplot as plt

    destination = Path(output_dir).resolve()
    featured = sorted(
        (item for item in records if item.get("featured_rank")),
        key=lambda item: int(item["featured_rank"]),
    )
    counts = Counter(str(item["condition_label"]) for item in records)
    cover, axis = plt.subplots(figsize=(8.27, 11.69), constrained_layout=True)
    axis.set_axis_off()
    axis.text(
        0.06,
        0.92,
        "SwarmWorld technology atlas",
        transform=axis.transAxes,
        fontsize=25,
        fontweight="bold",
        color="#17302A",
    )
    axis.text(
        0.06,
        0.875,
        f"{len(records):,} trace-derived technologies · {len(featured)} featured designs",
        transform=axis.transAxes,
        fontsize=12,
        color="#397263",
    )
    count_text = "\n".join(f"{label}: {count:,}" for label, count in sorted(counts.items()))
    method = (
        "Featured designs are visited in descending recorded lifetime peak performance. "
        "A candidate is retained only when its semantic cosine similarity to every already "
        "selected design is at or below "
        f"{float(selection_audit['maximum_allowed_cosine_similarity']):.2f}. Embeddings "
        "therefore enforce diversity and determine map position, but do not change the "
        "measured performance rank. At most "
        f"{selection_audit.get('maximum_per_semantic_cluster', 'unlimited')} featured "
        "designs are retained from one data-derived semantic cluster."
    )
    axis.text(
        0.06,
        0.76,
        "CORPUS\n" + count_text,
        transform=axis.transAxes,
        fontsize=10,
        linespacing=1.55,
        color="#20342E",
        va="top",
    )
    axis.text(
        0.06,
        0.53,
        "SELECTION METHOD\n" + "\n".join(textwrap.wrap(method, width=88)),
        transform=axis.transAxes,
        fontsize=10,
        linespacing=1.5,
        color="#20342E",
        va="top",
    )
    axis.text(
        0.06,
        0.26,
        "INTERPRETATION\n"
        "The map is a descriptive projection of recorded design language. Nearby points "
        "share semantic content; distance is not physical performance, causal influence, "
        "or proof of scientific validity.",
        transform=axis.transAxes,
        fontsize=10,
        linespacing=1.5,
        color="#20342E",
        va="top",
        wrap=True,
    )
    axis.text(
        0.06,
        0.08,
        DISCLAIMER,
        transform=axis.transAxes,
        fontsize=8.5,
        color="#7B4B37",
        va="bottom",
        wrap=True,
    )
    cover_path = destination / "technology-atlas-cover.pdf"
    cover.savefig(cover_path, facecolor="white")
    plt.close(cover)

    try:
        from pypdf import PdfWriter
    except ImportError as exc:
        raise RuntimeError(
            "the combined atlas PDF requires pypdf from the technology-images extra"
        ) from exc
    writer = PdfWriter()
    writer.append(str(cover_path), outline_item="Atlas overview")
    writer.append(
        str(destination / "technology-semantic-landscape.pdf"),
        outline_item="Semantic landscape",
    )
    writer.append(
        str(destination / "featured-technology-ranking.pdf"),
        outline_item="Featured ranking",
    )
    for item in featured:
        dossier_dir = item.get("dossier_dir")
        if not dossier_dir:
            continue
        sheet = destination / str(dossier_dir) / "briefing-sheet.pdf"
        if sheet.is_file():
            writer.append(str(sheet), outline_item=str(item.get("name", "Technology")))
    report = destination / "swarmworld-technology-atlas.pdf"
    with report.open("wb") as stream:
        writer.write(stream)
    return report


def _write_atlas_readme(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> None:
    featured = sorted(
        (item for item in records if item.get("featured_rank")),
        key=lambda item: int(item["featured_rank"]),
    )
    lines = [
        "# SwarmWorld technology atlas",
        "",
        f"This atlas contains **{len(records):,}** trace-derived technologies. The featured ",
        f"set contains **{len(featured)}** performance-ranked designs constrained to be ",
        "semantically distinct.",
        "",
        f"> {DISCLAIMER}",
        "",
        "## Open the explorer",
        "",
        "```bash",
        "python -m http.server 8765 --directory .",
        "```",
        "",
        "Then open `http://127.0.0.1:8765/technology-explorer.html`.",
        "",
        "## Featured technologies",
        "",
        "| Rank | Technology | Condition | N | Performance |",
        "|---:|---|---|---:|---:|",
    ]
    for item in featured:
        lines.append(
            f"| {item['featured_rank']} | {item['name']} | {item['condition_label']} | "
            f"{item['population_size']} | {float(item['performance']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            f"`{selection['selection_rule']}`. Ranking uses only "
            f"{selection['performance_measure']}; semantic embeddings only exclude "
            "near-duplicates and determine the atlas layout.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_technology_atlas(
    summary_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    conditions: Iterable[str] = CONDITION_ORDER,
    include_independent_members: bool = True,
    featured_count: int = 16,
    maximum_cosine_similarity: float = 0.82,
    maximum_per_cluster: int | None = 4,
    embedding_provider: str = "embeddinggemma",
    embedding_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
    embedding_dimensions: int = 768,
    embedding_batch_size: int = 16,
    generate_images: bool = False,
    image_model: str = DEFAULT_IMAGE_MODEL,
    image_size: str = "1536x1024",
    image_quality: str = "high",
    image_workers: int = 4,
    image_styles: Iterable[str] = DEFAULT_IMAGE_STYLES,
    overwrite_embeddings: bool = False,
    overwrite_images: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """Build the complete data atlas, paper figures, explorer, and top dossiers."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    technologies, collection_audit = collect_study_technologies_cached(
        summary_paths,
        destination,
        conditions=conditions,
        include_independent_members=include_independent_members,
    )
    if not technologies:
        raise ValueError("no technologies were found in the selected study summaries")
    embeddings, embedding_audit = build_semantic_embeddings(
        technologies,
        destination,
        provider=embedding_provider,
        model=embedding_model,
        dimensions=embedding_dimensions,
        batch_size=embedding_batch_size,
        overwrite=overwrite_embeddings,
    )
    coordinates, umap_audit = build_umap_embedding(
        embeddings, random_state=random_state, components=3
    )
    assignments, cluster_labels, cluster_audit = cluster_technologies(
        technologies, embeddings, random_state=random_state
    )
    selected, selection_audit = select_distinct_top_technologies(
        technologies,
        embeddings,
        count=featured_count,
        maximum_cosine_similarity=maximum_cosine_similarity,
        cluster_assignments=assignments,
        maximum_per_cluster=maximum_per_cluster,
    )
    figures = render_technology_atlas_figures(technologies, coordinates, selected, destination)
    folders, render_manifest = _generate_featured_dossiers(
        technologies,
        selected,
        destination,
        generate=generate_images,
        model=image_model,
        size=image_size,
        quality=image_quality,
        overwrite=overwrite_images,
        workers=image_workers,
        styles=tuple(image_styles),
    )

    _, performance_ranks = _performance_ranks(technologies)
    selected_ranks = {index: rank for rank, index in enumerate(selected, start=1)}
    records = [
        _catalog_record(
            technology,
            coordinates,
            index,
            cluster=int(assignments[index]),
            cluster_label=cluster_labels[int(assignments[index])],
            performance_rank=int(performance_ranks[index]),
            feature_rank=selected_ranks.get(index),
            dossier_dir=folders.get(index),
        )
        for index, technology in enumerate(technologies)
    ]
    records.sort(key=lambda item: int(item["performance_rank"]))
    _write_json(destination / "technology-catalog.json", records)
    _write_catalog_csv(destination / "technology-catalog.csv", records)
    explorer = write_threejs_explorer(records, destination)
    _write_atlas_readme(destination, records, selection_audit)
    report = build_technology_atlas_report(destination, records, selection_audit)

    audit = {
        "atlas_version": ATLAS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collection": collection_audit,
        "embedding": embedding_audit,
        "selection": selection_audit,
        "umap": umap_audit,
        "clustering": cluster_audit,
        "figures": figures,
        "explorer": str(explorer),
        "report": str(report),
        "render_manifest": str(destination / "featured-render-manifest.json"),
        "generated_image_count": sum(
            entry["status"] in {"generated", "existing"} for entry in render_manifest["entries"]
        ),
    }
    _write_json(destination / "technology-atlas-audit.json", audit)
    return audit


__all__ = [
    "ATLAS_VERSION",
    "CONDITION_ORDER",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LOCAL_EMBEDDING_MODEL",
    "build_semantic_embeddings",
    "build_technology_atlas",
    "build_technology_atlas_report",
    "build_umap_embedding",
    "cluster_technologies",
    "collect_study_technologies",
    "collect_study_technologies_cached",
    "render_technology_atlas_figures",
    "select_distinct_top_technologies",
    "technology_semantic_text",
    "write_threejs_explorer",
]
