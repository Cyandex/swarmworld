from __future__ import annotations

import json

import numpy as np

from biofoundry.cli import build_parser
from biofoundry.technology_atlas import (
    collect_study_technologies,
    select_distinct_top_technologies,
    technology_semantic_text,
    write_threejs_explorer,
)
from biofoundry.technology_visualization import (
    DISCLAIMER,
    build_technology_dossiers,
    build_technology_image_prompt,
    extract_technologies_from_trace,
    technology_controller_signature,
)


def _write_trace(tmp_path):
    path = tmp_path / "technology.jsonl"
    records = [
        {"type": "header", "metadata": {}},
        {
            "type": "event",
            "kind": "artifact_built",
            "tick": 12,
            "payload": {
                "agent": "agent_000001",
                "artifact_id": "artifact_00000001",
                "artifact_name": "Chitin Exchange Sheet",
                "tick": 12,
                "x": 3,
                "y": 4,
                "artifact_spec": {
                    "name": "Chitin Exchange Sheet",
                    "architecture": "Three aligned chitin layers with offset pores.",
                    "claimed_function": "Buffer moisture and retain connectivity.",
                    "bio_inspiration": ["arthropod cuticle", "stomata"],
                    "predicted_effects": ["moisture buffering"],
                    "geometry": {"layers": 3, "connectivity": 0.8},
                },
                "batch": {
                    "batch_id": "batch_1",
                    "composition": {"chitin": 0.9, "water": 0.1},
                    "properties": {"quality": 0.7},
                    "contributors": ["agent_000001"],
                    "causal_parents": ["test_1"],
                    "recipe": {
                        "inputs": [{"resource": "CHITIN", "mass": 0.3}],
                        "steps": [{"operation": "WASH", "intensity": 0.5}],
                        "output_form": "porous chitin sheet",
                        "design_principles": ["preserve exchange pathways"],
                    },
                },
                "program": {
                    "name": "passive_material_system",
                    "instructions": [{"op": "const", "dest": "r0", "value": 0}],
                },
            },
        },
        {
            "type": "event",
            "kind": "artifact_program_installed",
            "tick": 20,
            "payload": {
                "agent": "agent_000002",
                "artifact_id": "artifact_00000001",
                "program": {
                    "name": "Moisture Gate",
                    "instructions": [{"op": "sense"}, {"op": "set_open"}],
                },
                "lineage": {"parent_program_id": None},
            },
        },
        {
            "type": "snapshot",
            "snapshot": {
                "tick": 30,
                "artifacts": {
                    "count": 1,
                    "display_count": 1,
                    "ids": ["artifact_00000001"],
                    "name": ["Chitin Exchange Sheet"],
                    "creator": ["agent_000001"],
                    "created_tick": [12],
                    "architecture": ["Three aligned chitin layers with offset pores."],
                    "bio_inspiration": [["arthropod cuticle", "stomata"]],
                    "claimed_function": ["Buffer moisture and retain connectivity."],
                    "geometry": [{"layers": 3, "connectivity": 0.8}],
                    "lifetime_peak_performance": [0.73],
                    "lifetime_peak_services": {
                        "water_capture": [0.4],
                        "self_maintenance": [0.7],
                    },
                    "performance": [0.5],
                    "retired": [False],
                    "program": ["Moisture Gate"],
                    "program_id": ["program_1"],
                    "x": [3],
                    "y": [4],
                },
            },
        },
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return path


def test_extract_technology_preserves_build_and_snapshot_provenance(tmp_path):
    path = _write_trace(tmp_path)
    technologies = extract_technologies_from_trace(path)

    assert len(technologies) == 1
    technology = technologies[0]
    assert technology["name"] == "Chitin Exchange Sheet"
    assert technology["rank_performance"] == 0.73
    assert technology["recipe"]["steps"][0]["operation"] == "WASH"
    assert technology["lifetime_peak_services"]["self_maintenance"] == 0.7
    assert technology["program_history"][0]["program"]["name"] == "Moisture Gate"
    assert technology["source_trace_sha256"]


def test_prompts_are_dual_style_and_explicitly_non_evidentiary(tmp_path):
    technology = extract_technologies_from_trace(_write_trace(tmp_path))[0]
    engineering = build_technology_image_prompt(technology, "engineering")
    photorealistic = build_technology_image_prompt(technology, "photorealistic")
    overview = build_technology_image_prompt(technology, "overview")
    mechanism = build_technology_image_prompt(technology, "mechanism")

    assert "Three aligned chitin layers" in engineering
    assert "four balanced panels" in engineering
    assert "photorealistic laboratory-scale" in photorealistic
    assert "Do not invent numerical dimensions" in engineering
    assert "validated specimen" in photorealistic
    assert "exactly one isolated object" in overview
    assert "no title, letters, text" in overview
    assert "not a generic square porous tile" in mechanism
    assert "component specimens" in mechanism
    assert "chitin 90%" in mechanism
    assert "simulator evidence" in DISCLAIMER

    controller = technology_controller_signature(technology)
    assert controller["name"] == "Moisture Gate"
    assert controller["actuators"] == ("set_open",)


def test_cli_dossier_generation_requires_explicit_flag():
    args = build_parser().parse_args(
        ["technology-dossiers", "trace.jsonl", "--output-dir", "dossiers"]
    )
    assert args.generate is False
    assert args.model == "gpt-image-2"
    assert args.styles == ["engineering", "photorealistic"]

    overview_args = build_parser().parse_args(
        [
            "technology-dossiers",
            "trace.jsonl",
            "--output-dir",
            "dossiers",
            "--styles",
            "overview",
        ]
    )
    assert overview_args.styles == ["overview"]

    mechanism_args = build_parser().parse_args(
        [
            "technology-dossiers",
            "trace.jsonl",
            "--output-dir",
            "dossiers",
            "--styles",
            "mechanism",
        ]
    )
    assert mechanism_args.styles == ["mechanism"]


def test_cli_atlas_defaults_to_distinct_top_16_and_no_image_calls():
    args = build_parser().parse_args(["technology-atlas", "summary.json", "--output-dir", "atlas"])

    assert args.featured_count == 16
    assert args.maximum_cosine_similarity == 0.82
    assert args.maximum_per_cluster == 4
    assert args.embedding_provider == "embeddinggemma"
    assert args.embedding_model == "google/embeddinggemma-300m"
    assert args.image_workers == 4
    assert args.image_styles == ["engineering", "photorealistic"]
    assert args.generate is False

    overview_args = build_parser().parse_args(
        [
            "technology-atlas",
            "summary.json",
            "--output-dir",
            "atlas",
            "--image-styles",
            "overview",
        ]
    )
    assert overview_args.image_styles == ["overview"]

    mechanism_args = build_parser().parse_args(
        [
            "technology-atlas",
            "summary.json",
            "--output-dir",
            "atlas",
            "--image-styles",
            "mechanism",
        ]
    )
    assert mechanism_args.image_styles == ["mechanism"]


def test_collect_study_technologies_assigns_globally_unique_provenance(tmp_path):
    trace = _write_trace(tmp_path)
    summary = tmp_path / "study-summary.json"
    summary.write_text(
        json.dumps(
            [
                {
                    "condition": "full",
                    "population_size": 50,
                    "seed": 3201,
                    "output": str(trace),
                    "technology_ecology": {"artifacts": []},
                }
            ]
        ),
        encoding="utf-8",
    )

    technologies, audit = collect_study_technologies([summary], conditions=["full"])

    assert audit["technology_count"] == 1
    assert technologies[0]["technology_uid"].startswith("full-n50-s3201-")
    assert technologies[0]["condition"] == "full"
    semantic_text = technology_semantic_text(technologies[0])
    assert "Three aligned chitin layers" in semantic_text
    assert "Buffer moisture" in semantic_text


def test_distinct_selection_rejects_semantic_near_duplicates():
    technologies = [
        {"technology_uid": "best", "rank_performance": 1.0, "name": "A"},
        {"technology_uid": "duplicate", "rank_performance": 0.9, "name": "B"},
        {"technology_uid": "different", "rank_performance": 0.8, "name": "C"},
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    selected, audit = select_distinct_top_technologies(
        technologies,
        embeddings,
        count=2,
        maximum_cosine_similarity=0.82,
    )

    assert selected == [0, 2]
    assert audit["selected_global_performance_ranks"] == [1, 3]


def test_distinct_selection_enforces_cluster_coverage_cap():
    technologies = [
        {"technology_uid": "a", "rank_performance": 1.0, "name": "A"},
        {"technology_uid": "b", "rank_performance": 0.9, "name": "B"},
        {"technology_uid": "c", "rank_performance": 0.8, "name": "C"},
    ]
    embeddings = np.eye(3, dtype=np.float32)

    selected, audit = select_distinct_top_technologies(
        technologies,
        embeddings,
        count=2,
        maximum_cosine_similarity=0.99,
        cluster_assignments=np.asarray([0, 0, 1]),
        maximum_per_cluster=1,
    )

    assert selected == [0, 2]
    assert audit["selected_cluster_counts"] == {"0": 1, "1": 1}


def test_threejs_explorer_is_python_generated_and_data_driven(tmp_path):
    records = [
        {
            "technology_uid": "technology-1",
            "name": "Porous sheet",
            "condition": "full",
            "condition_label": "Full",
            "population_size": 50,
            "seed": 1,
            "creator": "agent_1",
            "created_tick": 10,
            "performance": 0.7,
            "performance_rank": 1,
            "featured_rank": 1,
            "cluster_label": "porous · moisture",
            "umap": [0.0, 1.0, 2.0],
            "architecture": "A porous sheet.",
            "claimed_function": "Capture water.",
            "explanation": "A porous sheet captures water.",
            "bio_inspiration": ["leaf"],
            "composition": {},
            "recipe": {},
            "services": {},
            "source_trace": "trace.jsonl",
            "source_trace_sha256": "abc",
            "dossier_dir": "featured/01",
            "engineering_image": "featured/01/engineering.png",
            "photorealistic_image": "featured/01/photorealistic.png",
        }
    ]

    explorer = write_threejs_explorer(records, tmp_path)

    assert explorer.is_file()
    explorer_text = explorer.read_text(encoding="utf-8")
    assert "OrbitControls" in explorer_text
    vendor_core = tmp_path / "vendor/three.core.js"
    vendor_controls = tmp_path / "vendor/controls/OrbitControls.js"
    if vendor_core.exists() or vendor_controls.exists():
        assert vendor_core.is_file()
        assert vendor_controls.is_file()
    else:
        assert "https://cdn.jsdelivr.net/npm/three@" in explorer_text
    data = json.loads((tmp_path / "technology-explorer-data.json").read_text())
    assert data[0]["technology_uid"] == "technology-1"


def test_offline_dossier_build_is_portable_and_writes_all_sheet_formats(tmp_path):
    output = tmp_path / "dossiers"
    manifest = build_technology_dossiers(_write_trace(tmp_path), output, top_k=1)

    assert manifest["generated"] is False
    assert manifest["technology_count"] == 1
    assert {entry["status"] for entry in manifest["entries"]} == {"prompt_only"}
    assert all(not entry["output"].startswith("/") for entry in manifest["entries"])
    dossier = next(path.parent for path in output.rglob("technology.json"))
    assert (dossier / "prompts.json").is_file()
    assert (dossier / "briefing-sheet.png").is_file()
    assert (dossier / "briefing-sheet.pdf").is_file()
    assert (dossier / "briefing-sheet.svg").is_file()
