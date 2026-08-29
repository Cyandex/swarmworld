"""Trace-grounded technology catalogs, image prompts, and briefing sheets.

The generated pictures are explanatory concept renderings.  They are never used as
simulation evidence, and the pipeline does not infer dimensions, measured properties,
or fabrication claims that are absent from the recorded trace.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import textwrap
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .events import read_records

TECHNOLOGY_DOSSIER_VERSION = 1
DEFAULT_IMAGE_MODEL = "gpt-image-2"
IMAGE_STYLES = ("engineering", "photorealistic", "overview", "mechanism")
DEFAULT_IMAGE_STYLES = ("engineering", "photorealistic")
DISCLAIMER = (
    "AI-generated explanatory concept rendering grounded in a SwarmWorld trace. "
    "It is not simulator evidence, a measured specimen, a fabrication drawing, "
    "or a patent claim."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:72] or "technology"


def _indexed(value: Any, index: int, count: int) -> Any:
    """Read one row from a recursively columnar snapshot payload."""

    if isinstance(value, Mapping):
        return {key: _indexed(item, index, count) for key, item in value.items()}
    if isinstance(value, list) and len(value) == count:
        return value[index]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _merge_snapshot_artifacts(
    catalog: dict[str, dict[str, Any]],
    snapshot: Mapping[str, Any],
) -> None:
    table = dict(snapshot.get("artifacts", {}))
    ids = list(table.get("ids", []))
    count = len(ids)
    for index, artifact_id in enumerate(ids):
        artifact_key = str(artifact_id)
        row = {
            key: _indexed(value, index, count)
            for key, value in table.items()
            if key not in {"count", "display_count", "ids"}
        }
        row["artifact_id"] = artifact_key
        catalog.setdefault(artifact_key, {}).update(_jsonable(row))


def extract_technologies_from_trace(
    trace_path: str | Path,
    *,
    include_retired: bool = True,
    minimum_peak_performance: float = 0.0,
    top_k: int | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Extract all built artifacts and their design provenance from a trace.

    ``artifact_built`` events retain the construction recipe and initial controller;
    the last snapshot adds final and lifetime performance, services, health, and the
    installed controller.  Sorting is deterministic.
    """

    source = Path(trace_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    catalog: dict[str, dict[str, Any]] = {}
    program_history: dict[str, list[dict[str, Any]]] = {}
    last_snapshot: dict[str, Any] | None = None
    records = read_records(source)
    for record in tqdm(
        records,
        desc="Reading technology trace",
        unit="record",
        disable=not show_progress,
    ):
        record_type = str(record.get("type", ""))
        if record_type == "snapshot":
            last_snapshot = dict(record.get("snapshot", {}))
            continue
        if record_type != "event":
            continue
        kind = str(record.get("kind", ""))
        payload = dict(record.get("payload", {}))
        if kind == "artifact_built":
            artifact_id = str(payload.get("artifact_id", ""))
            if not artifact_id:
                continue
            spec = dict(payload.get("artifact_spec", {}))
            batch = dict(payload.get("batch", {}))
            program = dict(payload.get("program", {}))
            catalog[artifact_id] = {
                "artifact_id": artifact_id,
                "name": str(payload.get("artifact_name") or spec.get("name") or artifact_id),
                "creator": str(payload.get("agent", "")),
                "created_tick": int(payload.get("tick", record.get("tick", 0))),
                "x": payload.get("x"),
                "y": payload.get("y"),
                "architecture": str(spec.get("architecture", "")),
                "bio_inspiration": list(spec.get("bio_inspiration", [])),
                "claimed_function": str(spec.get("claimed_function", "")),
                "predicted_effects": list(spec.get("predicted_effects", [])),
                "geometry": _jsonable(spec.get("geometry", {})),
                "batch_id": batch.get("batch_id"),
                "composition": _jsonable(batch.get("composition", {})),
                "material_properties": _jsonable(batch.get("properties", {})),
                "recipe": _jsonable(batch.get("recipe", {})),
                "contributors": list(batch.get("contributors", [])),
                "causal_parents": list(batch.get("causal_parents", [])),
                "initial_program": _jsonable(program),
            }
        elif kind == "artifact_program_installed":
            artifact_id = str(payload.get("artifact_id", ""))
            if artifact_id:
                program_history.setdefault(artifact_id, []).append(
                    {
                        "tick": int(record.get("tick", 0)),
                        "agent": payload.get("agent"),
                        "program": _jsonable(payload.get("program", {})),
                        "lineage": _jsonable(payload.get("lineage", {})),
                    }
                )

    if last_snapshot is not None:
        _merge_snapshot_artifacts(catalog, last_snapshot)
    for artifact_id, history in program_history.items():
        catalog.setdefault(artifact_id, {"artifact_id": artifact_id})["program_history"] = history

    source_sha256 = _sha256_file(source)
    result: list[dict[str, Any]] = []
    for artifact in catalog.values():
        artifact["source_trace"] = str(source)
        artifact["source_trace_sha256"] = source_sha256
        artifact["trace_final_tick"] = int((last_snapshot or {}).get("tick", 0))
        retired = bool(artifact.get("retired", False))
        peak = float(
            artifact.get("lifetime_peak_performance")
            or artifact.get("peak_performance")
            or artifact.get("performance")
            or 0.0
        )
        artifact["rank_performance"] = peak
        if (include_retired or not retired) and peak >= minimum_peak_performance:
            result.append(_jsonable(artifact))
    result.sort(
        key=lambda item: (
            -float(item.get("rank_performance", 0.0)),
            str(item.get("artifact_id", "")),
        )
    )
    if top_k is not None:
        result = result[:top_k]
    return result


def _compact_mapping(value: Any, *, limit: int = 12) -> str:
    if not isinstance(value, Mapping):
        return str(value or "not recorded")
    pairs = []
    for key, item in list(value.items())[:limit]:
        if isinstance(item, (int, float)):
            pairs.append(f"{key}={float(item):.3g}")
        else:
            pairs.append(f"{key}={item}")
    return ", ".join(pairs) or "not recorded"


def _recipe_text(recipe: Any) -> str:
    if not isinstance(recipe, Mapping):
        return "No construction recipe was retained."
    inputs = recipe.get("inputs", [])
    input_text = ", ".join(
        f"{item.get('resource', '?')} ({item.get('mass', '?')} mass)"
        for item in inputs
        if isinstance(item, Mapping)
    )
    steps = recipe.get("steps", [])
    step_text = " → ".join(
        f"{item.get('operation', '?')} at intensity {item.get('intensity', '?')}"
        for item in steps
        if isinstance(item, Mapping)
    )
    principles = ", ".join(map(str, recipe.get("design_principles", [])))
    return (
        f"Inputs: {input_text or 'not recorded'}. Process: {step_text or 'not recorded'}. "
        f"Output form: {recipe.get('output_form', 'not recorded')}. "
        f"Design principles: {principles or 'not recorded'}."
    )


def _program_text(technology: Mapping[str, Any]) -> str:
    signature = technology_controller_signature(technology)
    operations = signature["operations"]
    return (
        f"Controller: {signature['name']}. Recorded operations: {', '.join(operations) or 'none'}."
    )


def technology_controller_signature(technology: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, trace-derived controller signature for display and prompts."""

    name = technology.get("program")
    initial = technology.get("initial_program", {})
    if not name and isinstance(initial, Mapping):
        name = initial.get("name")
    history = technology.get("program_history", [])
    program = None
    if history and isinstance(history, list) and isinstance(history[-1], Mapping):
        program = history[-1].get("program")
    if not isinstance(program, Mapping):
        program = initial if isinstance(initial, Mapping) else {}
    instructions = program.get("instructions", []) if isinstance(program, Mapping) else []
    operations = tuple(
        str(item.get("op", ""))
        for item in instructions
        if isinstance(item, Mapping) and item.get("op")
    )
    sensors = tuple(
        dict.fromkeys(
            str(item.get("sensor"))
            for item in instructions
            if isinstance(item, Mapping) and item.get("op") == "sense" and item.get("sensor")
        )
    )
    actuator_names = {
        "collect_water",
        "emit_signal",
        "grow",
        "heal",
        "reduce_contamination",
        "set_open",
    }
    actuators = tuple(
        dict.fromkeys(operation for operation in operations if operation in actuator_names)
    )
    return {
        "name": str(name or program.get("name") or "passive material system"),
        "operations": operations,
        "sensors": sensors,
        "actuators": actuators,
    }


def _mechanism_material_brief(technology: Mapping[str, Any]) -> str:
    composition = technology.get("composition")
    if not isinstance(composition, Mapping):
        return "No component hierarchy was recorded."
    material_appearance = {
        "cellulose": "pale straw-colored directional fibers",
        "chitin": "ivory laminated fibers or shell-like ribs",
        "lignin": "warm-brown stiffening phase",
        "mineral": "gray mineral grains, plates, or load-bearing ribs",
        "protein": "subtle translucent amber binder",
        "water": "muted cyan hydrated pores or channels",
    }
    ranked = sorted(
        (
            (str(name), float(value))
            for name, value in composition.items()
            if isinstance(value, (int, float)) and float(value) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )[:4]
    details = [
        f"{name} {100.0 * fraction:.0f}% as {material_appearance.get(name, name)}"
        for name, fraction in ranked
    ]
    return "Visible recorded component hierarchy: " + "; ".join(details) + "."


def build_technology_image_prompt(
    technology: Mapping[str, Any],
    style: str,
) -> str:
    """Create a trace-grounded prompt for one explanatory concept rendering."""

    if style not in IMAGE_STYLES:
        raise ValueError(f"style must be one of {IMAGE_STYLES}")
    name = str(technology.get("name") or technology.get("artifact_id") or "Technology")
    bio = ", ".join(map(str, technology.get("bio_inspiration", []))) or "not recorded"
    architecture = technology.get("architecture") or "not recorded"
    claimed_function = technology.get("claimed_function") or "not recorded"
    recipe = _recipe_text(technology.get("recipe"))
    common = (
        f"Create a scientifically sober concept visualization of the SwarmWorld invention "
        f"named '{name}'. Recorded architecture: {architecture}. "
        f"Recorded claimed function: {claimed_function}. "
        f"Recorded biological inspirations: {bio}. Recorded geometry: "
        f"{_compact_mapping(technology.get('geometry'))}. Recorded composition: "
        f"{_compact_mapping(technology.get('composition'))}. {recipe} "
        f"{_program_text(technology)} Use only these recorded concepts. Do not invent numerical "
        "dimensions, measured performance, organisms, components, or validation claims. "
    )
    if style == "overview":
        return common + (
            "Create one high-resolution publication-gallery portrait of the complete technology. "
            "Show exactly one isolated object in a wide 16:10 composition, centered with generous "
            "white margins and fully visible on a pure white background. Use a consistent "
            "three-quarter isometric camera, soft neutral studio illumination, crisp material "
            "detail, subtle contact shadow, restrained bone, mineral, cellulose, fungal, and "
            "muted teal material tones, and "
            "scientifically plausible structure. Emphasize the recorded architecture and internal "
            "organization while preserving a clean silhouette at small journal-figure scale. "
            "Include no title, letters, text, labels, callout lines, arrows, legend, border, "
            "multiple panels, "
            "cutaway inset, people, tools, laboratory scene, watermark, or decorative background."
        )
    if style == "mechanism":
        return (
            common
            + _mechanism_material_brief(technology)
            + (
                " Create one high-resolution publication-gallery mechanism portrait that is "
                "visually specific to this recorded invention, not a generic square porous tile. "
                "Show one main complete technology with an architecture-specific silhouette and a "
                "clean partial cutaway or gently separated edge that exposes its recorded layers, "
                "fibers, channels, junctions, and material phases. Beside the main object, arrange "
                "two to four small, orderly, unlabeled component specimens derived only from the "
                "recorded composition and recipe inputs. Make thin veils and membranes visibly "
                "thin and flexible; woven trellises open and branching; laminated panels layered "
                "and directional; mineral lattices rigid and ribbed; and compartmental interfaces "
                "visibly partitioned, but only when those features are supported by the recorded "
                "architecture. Use a wide 16:10 composition, a "
                "three-quarter isometric camera, pure white background, generous margins, crisp "
                "scientific material detail, and a subtle contact shadow. Use distinct component "
                "textures and the recorded relative component prominence so the sixteen inventions "
                "can be compared visually. Include no title, letters, text, labels, arrows, "
                "legend, border, multiple panels, people, tools, laboratory scene, watermark, "
                "or decorative background."
            )
        )
    if style == "engineering":
        return common + (
            "Render a clean white-background engineering concept sheet with four balanced panels: "
            "A overall architecture, B exploded material layers or constituent organization, "
            "C functional cross-section showing qualitative transport/load/control pathways, "
            "and D controller or adaptation sequence. Use precise black and muted teal linework, "
            "subtle material color accents, consistent scale across related views, thin callout "
            "leaders, and only the panel letters A–D plus a short exact title. Avoid patent "
            "numbers, "
            "logos, dense prose, fake measurements, decorative borders, or legal language. "
            "The result should look like a high-end scientific technical illustration, "
            "not a patent claim."
        )
    return common + (
        "Render a photorealistic laboratory-scale concept prototype on a neutral light-gray bench, "
        "with a clean cutaway or neighboring section that reveals the recorded internal "
        "architecture. "
        "Use physically plausible material texture and wet/dry state, restrained studio lighting, "
        "natural depth of field, and no people. Include no labels, text, logos, rulers, fake "
        "instruments, or unrecorded biological organisms. Make it unmistakably a concept "
        "rendering, not a photograph "
        "of a validated specimen."
    )


def _top_services(technology: Mapping[str, Any]) -> list[tuple[str, float]]:
    candidates = technology.get("lifetime_peak_services") or technology.get("peak_services")
    if not isinstance(candidates, Mapping):
        candidates = technology.get("services", {})
    values = []
    if isinstance(candidates, Mapping):
        for key, value in candidates.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append((str(key), float(value)))
    return sorted(values, key=lambda item: (-item[1], item[0]))[:6]


def _wrapped(value: Any, width: int) -> str:
    return textwrap.fill(str(value or "Not recorded."), width=width)


def render_technology_briefing_sheet(
    technology: Mapping[str, Any],
    image_paths: Mapping[str, Path],
    output_stem: str | Path,
) -> dict[str, str]:
    """Render a deterministic, paper-ready one-page technology dossier."""

    cache_dir = Path(tempfile.gettempdir()) / "biofoundry-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib.pyplot as plt

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(11.7, 8.3), facecolor="white", constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(0.17, 1.0, 0.82))
    title_axis = figure.add_subplot(grid[0, :])
    overview_only = (
        image_paths.get("overview") is not None
        and image_paths.get("engineering") is None
        and image_paths.get("photorealistic") is None
    )
    if overview_only:
        image_axes = [(figure.add_subplot(grid[1, :]), "overview", "A  Overview concept")]
    else:
        image_axes = [
            (figure.add_subplot(grid[1, :2]), "engineering", "A  Engineering concept"),
            (figure.add_subplot(grid[1, 2:]), "photorealistic", "B  Photorealistic concept"),
        ]
    text_axis = figure.add_subplot(grid[2, :2])
    metrics_axis = figure.add_subplot(grid[2, 2:])
    for axis in (title_axis, *(item[0] for item in image_axes), text_axis, metrics_axis):
        axis.set_axis_off()

    name = str(technology.get("name") or technology.get("artifact_id"))
    title_size = 17 if len(name) <= 34 else 14 if len(name) <= 50 else 12
    title_axis.text(
        0,
        0.78,
        name,
        fontsize=title_size,
        fontweight="bold",
        color="#17302a",
    )
    title_axis.text(
        0,
        0.2,
        (
            f"{technology.get('artifact_id')}  ·  creator {technology.get('creator', '?')}  ·  "
            f"tick {technology.get('created_tick', '?')}  ·  peak performance "
            f"{float(technology.get('rank_performance', 0.0)):.3f}"
            + (
                f"  ·  {technology.get('condition_label', technology.get('condition'))} "
                f"N={technology.get('population_size')} seed={technology.get('seed')}"
                if technology.get("condition")
                else ""
            )
        ),
        fontsize=9,
        color="#526b75",
    )

    for axis, style, label in image_axes:
        path = image_paths.get(style)
        if path is not None and path.exists():
            axis.imshow(plt.imread(path))
        else:
            axis.set_facecolor("#f3f6f4")
            axis.text(
                0.5,
                0.5,
                f"{style.title()} image\nnot generated",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=11,
                color="#6f7f79",
            )
        axis.set_axis_off()
        axis.set_title(label, loc="left", fontsize=10, fontweight="bold", color="#17302a")

    bio = ", ".join(map(str, technology.get("bio_inspiration", []))) or "Not recorded"
    body = (
        "RECORDED DESIGN\n"
        f"{_wrapped(technology.get('architecture'), 70)}\n\n"
        "CLAIMED FUNCTION\n"
        f"{_wrapped(technology.get('claimed_function'), 70)}\n\n"
        "BIOLOGICAL INSPIRATION\n"
        f"{_wrapped(bio, 70)}\n\n"
        "CONSTRUCTION\n"
        f"{_wrapped(_recipe_text(technology.get('recipe')), 70)}\n\n"
        "CONTROL\n"
        f"{_wrapped(_program_text(technology), 70)}"
    )
    text_axis.text(0, 1, body, va="top", fontsize=7.4, linespacing=1.22, color="#20342e")

    services = _top_services(technology)
    if services:
        chart = metrics_axis.inset_axes([0.02, 0.50, 0.96, 0.45])
        names = [name.replace("_", " ") for name, _ in services][::-1]
        values = [value for _, value in services][::-1]
        positions = list(range(len(names)))
        chart.barh(positions, values, color="#2f8f73")
        chart.set_yticks(positions, names)
        chart.set_title("Recorded peak service profile", loc="left", fontsize=10, fontweight="bold")
        chart.spines[["top", "right"]].set_visible(False)
        chart.tick_params(labelsize=8)
    metrics_axis.text(
        0.02,
        0.39,
        f"Geometry: {_wrapped(_compact_mapping(technology.get('geometry')), 82)}",
        transform=metrics_axis.transAxes,
        va="top",
        fontsize=8,
        color="#20342e",
    )
    metrics_axis.text(
        0.02,
        0.14,
        DISCLAIMER,
        transform=metrics_axis.transAxes,
        va="top",
        fontsize=7.5,
        color="#7b4b37",
        wrap=True,
    )

    outputs: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs[suffix] = str(path.resolve())
    plt.close(figure)
    return outputs


def _write_markdown_dossier(
    technology: Mapping[str, Any],
    image_paths: Mapping[str, Path],
    briefing_paths: Mapping[str, str],
    destination: Path,
) -> None:
    name = str(technology.get("name") or technology.get("artifact_id"))
    bio = ", ".join(map(str, technology.get("bio_inspiration", []))) or "Not recorded"
    lines = [
        f"# {name}",
        "",
        f"- Artifact: `{technology.get('artifact_id')}`",
        f"- Creator: `{technology.get('creator', '')}`",
        f"- Created: tick {technology.get('created_tick', '')}",
        "- Recorded lifetime peak performance: "
        f"{float(technology.get('rank_performance', 0.0)):.4f}",
        f"- Source trace: `{technology.get('source_trace')}`",
        "",
    ]
    for style in IMAGE_STYLES:
        path = image_paths.get(style)
        if path is not None and path.exists():
            lines.extend([f"![{style} concept]({path.name})", ""])
    lines.extend(
        [
            "## Recorded design",
            "",
            str(technology.get("architecture") or "Not recorded."),
            "",
            "## Claimed function",
            "",
            str(technology.get("claimed_function") or "Not recorded."),
            "",
            "## Biological inspiration",
            "",
            bio,
            "",
            "## Construction and control",
            "",
            _recipe_text(technology.get("recipe")),
            "",
            _program_text(technology),
            "",
            "## Reproducibility",
            "",
            "Briefing sheet: "
            f"[{Path(briefing_paths['pdf']).name}]"
            f"({Path(briefing_paths['pdf']).name})",
            "",
            f"> {DISCLAIMER}",
            "",
        ]
    )
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _generate_image(
    *,
    client: Any,
    model: str,
    prompt: str,
    size: str,
    quality: str,
    output: Path,
) -> None:
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
    )
    data = response.data[0]
    encoded = getattr(data, "b64_json", None)
    if not encoded:
        raise RuntimeError("image response did not contain b64_json")
    output.write_bytes(base64.b64decode(encoded))


def build_technology_dossiers(
    trace_path: str | Path,
    output_dir: str | Path,
    *,
    styles: Iterable[str] = DEFAULT_IMAGE_STYLES,
    generate: bool = False,
    model: str = DEFAULT_IMAGE_MODEL,
    size: str = "1536x1024",
    quality: str = "high",
    top_k: int | None = None,
    include_retired: bool = True,
    minimum_peak_performance: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build prompts and dossiers for every selected trace-derived technology.

    Network calls happen only when ``generate=True``.  Without it, the exact catalog,
    prompts, metadata, and placeholder briefing sheets are still generated.
    """

    selected_styles = tuple(dict.fromkeys(styles))
    unknown = set(selected_styles) - set(IMAGE_STYLES)
    if unknown:
        raise ValueError(f"unknown image styles: {sorted(unknown)}")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    technologies = extract_technologies_from_trace(
        trace_path,
        include_retired=include_retired,
        minimum_peak_performance=minimum_peak_performance,
        top_k=top_k,
    )
    client = None
    if generate:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Image generation requires the optional 'technology-images' dependencies"
            ) from exc
        client = OpenAI()

    manifest_entries: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    progress = tqdm(technologies, desc="Technology dossiers", unit="technology")
    for rank, technology in enumerate(progress, start=1):
        artifact_id = str(technology.get("artifact_id"))
        name = str(technology.get("name") or artifact_id)
        progress.set_postfix_str(name[:30])
        folder = destination / f"{rank:03d}-{_slug(name)}-{_slug(artifact_id)}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "technology.json").write_text(
            json.dumps(technology, indent=2, sort_keys=True), encoding="utf-8"
        )
        prompts: dict[str, str] = {}
        image_paths: dict[str, Path] = {}
        for style in selected_styles:
            prompt = build_technology_image_prompt(technology, style)
            prompts[style] = prompt
            image_path = folder / f"{style}.png"
            status = "prompt_only"
            error = None
            if generate and (overwrite or not image_path.exists()):
                try:
                    _generate_image(
                        client=client,
                        model=model,
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        output=image_path,
                    )
                    status = "generated"
                except Exception as exc:  # preserve the rest of a long batch
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
            elif image_path.exists():
                status = "existing"
            if image_path.exists():
                image_paths[style] = image_path
            manifest_entries.append(
                {
                    "artifact_id": artifact_id,
                    "name": name,
                    "rank": rank,
                    "style": style,
                    "model": model,
                    "size": size,
                    "quality": quality,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "output": str(image_path.relative_to(destination)),
                    "status": status,
                    "error": error,
                }
            )
        (folder / "prompts.json").write_text(
            json.dumps(prompts, indent=2, sort_keys=True), encoding="utf-8"
        )
        briefing_paths = render_technology_briefing_sheet(
            technology,
            image_paths,
            folder / "briefing-sheet",
        )
        _write_markdown_dossier(
            technology,
            image_paths,
            briefing_paths,
            folder,
        )
        index_rows.append(
            {
                "rank": rank,
                "artifact_id": artifact_id,
                "name": name,
                "creator": technology.get("creator"),
                "created_tick": technology.get("created_tick"),
                "peak_performance": technology.get("rank_performance"),
                "retired": technology.get("retired", False),
                "dossier": str((folder / "README.md").relative_to(destination)),
                "briefing_pdf": str(Path(briefing_paths["pdf"]).relative_to(destination)),
            }
        )

    manifest = {
        "version": TECHNOLOGY_DOSSIER_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_trace": str(Path(trace_path)),
        "source_trace_sha256": _sha256_file(Path(trace_path)),
        "model": model,
        "styles": list(selected_styles),
        "generated": generate,
        "technology_count": len(technologies),
        "disclaimer": DISCLAIMER,
        "entries": manifest_entries,
    }
    (destination / "technology-render-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if index_rows:
        with (destination / "technology-index.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
            writer.writeheader()
            writer.writerows(index_rows)
    index_lines = [
        "# Trace-derived technology dossiers",
        "",
        f"Source: `{Path(trace_path)}`",
        "",
        f"> {DISCLAIMER}",
        "",
        "| Rank | Technology | Peak performance | Dossier |",
        "|---:|---|---:|---|",
    ]
    for row in index_rows:
        index_lines.append(
            f"| {row['rank']} | {row['name']} | {float(row['peak_performance']):.4f} | "
            f"[open]({row['dossier']}) |"
        )
    (destination / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "DEFAULT_IMAGE_MODEL",
    "DISCLAIMER",
    "IMAGE_STYLES",
    "build_technology_dossiers",
    "build_technology_image_prompt",
    "extract_technologies_from_trace",
    "render_technology_briefing_sheet",
]
