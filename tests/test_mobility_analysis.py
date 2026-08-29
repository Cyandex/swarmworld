from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from biofoundry.cli import build_parser
from biofoundry.mobility_analysis import (
    _build_behavior_embedding,
    _build_regime_embedding,
    _render_self_organization,
    extract_spatial_trace,
    gini,
)


def _snapshot(tick, x, y, distance, visited):
    return {
        "type": "snapshot",
        "snapshot": {
            "tick": tick,
            "agents": {
                "count": 2,
                "display_count": 2,
                "ids": ["agent_0", "agent_1"],
                "x": x,
                "y": y,
                "distance_traveled": distance,
                "distinct_cells_visited": visited,
            },
            "artifacts": {
                "count": 1,
                "display_count": 1,
                "x": [2],
                "y": [2],
                "retired": [False],
            },
            "world": {
                "width": 4,
                "height": 4,
                "terrain": [[1, 1, 1, 1] for _ in range(4)],
                "resource_mass": [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
            },
        },
    }


def test_extract_spatial_trace_uses_full_population_and_tracks_relocation(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = [
        {"type": "header", "metadata": {}},
        _snapshot(0, [0, 3], [0, 3], [0.0, 0.0], [1, 1]),
        {
            "type": "event",
            "tick": 5,
            "kind": "action_result",
            "payload": {
                "agent": "agent_0",
                "success": True,
                "verb": 8,
            },
        },
        {
            "type": "event",
            "tick": 5,
            "kind": "artifact_built",
            "payload": {
                "agent": "agent_0",
                "artifact_id": "artifact_0",
                "batch": {"contributors": ["agent_1"]},
            },
        },
        {
            "type": "event",
            "tick": 6,
            "kind": "message_delivered",
            "payload": {
                "sender": "agent_0",
                "recipients": ["agent_1"],
            },
        },
        _snapshot(10, [1, 2], [1, 2], [2.0, 2.0], [3, 3]),
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")

    report = extract_spatial_trace(path, proximity_radius=1.5, retain_positions=True)

    assert [point["tick"] for point in report["points"]] == [0, 10]
    assert report["points"][1]["relocating_agent_fraction"] == 1.0
    assert report["points"][1]["mean_distance_traveled"] == 2.0
    assert report["points"][1]["spatial_entropy"] == pytest.approx(1.0)
    assert report["summary"]["encounter_exposure_auc"] == pytest.approx(0.5)
    assert report["summary"]["artifact_contact_auc"] == pytest.approx(0.75)
    assert report["summary"]["artifact_bound_movement_auc"] == pytest.approx(0.5)
    assert report["summary"]["artifact_work_events_per_100_agent_ticks"] == 5.0
    assert report["summary"]["explicit_interactions_per_100_agent_ticks"] == 5.0
    assert len(report["trajectories"]["agent_0"]) == 2
    assert report["details"]["artifact_participants"]["artifact_0"] == [
        "agent_0",
        "agent_1",
    ]
    agent_0 = next(row for row in report["agent_rows"] if row["agent_id"] == "agent_0")
    assert agent_0["movement_snapshot_fraction"] == 1.0
    assert agent_0["net_displacement"] == pytest.approx(2**0.5)
    assert agent_0["moving_toward_artifact_fraction"] == 1.0
    assert agent_0["artifact_work_rate_per_100_ticks"] == 10.0
    assert agent_0["social_interaction_rate_per_100_ticks"] == 10.0
    assert agent_0["unique_social_partners"] == 1
    assert report["transition_counts"][0][3] == 1
    assert report["transition_counts"][2][3] == 1


def test_gini_is_zero_for_equal_paths_and_positive_for_unequal_paths():
    assert gini([2, 2, 2]) == 0.0
    assert gini([0, 0, 3]) == pytest.approx(2 / 3)


def test_cli_mobility_defaults_are_explicit():
    args = build_parser().parse_args(
        ["analyze-mobility", "summary.json", "--output-dir", "mobility"]
    )
    assert args.bootstrap_resamples == 20_000
    assert args.artifact_proximity_radius == 3.0
    assert args.encounter_radius == 2.0


def test_behavior_embedding_is_trace_feature_only_and_writes_seed_fractions(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)
    rows = []
    for index in range(24):
        mobile = index >= 12
        rows.append(
            {
                "condition": "full" if index % 2 else "no-communication",
                "population_size": 12,
                "seed": 1 + index // 12,
                "agent_id": f"agent_{index}",
                "distance_traveled": 3.0 + 30.0 * mobile + index / 10,
                "net_displacement": 1.0 + 8.0 * mobile,
                "displacement_efficiency": 0.2 + 0.4 * mobile,
                "distinct_coarse_cells": 1 + 8 * mobile,
                "movement_snapshot_fraction": 0.05 + 0.7 * mobile,
                "artifact_contact_fraction": 0.8 - 0.5 * mobile,
                "artifact_bound_movement_fraction": 0.1 + 0.2 * mobile,
                "moving_toward_artifact_fraction": 0.2 + 0.4 * mobile,
                "encounter_exposure_per_snapshot": 0.1 + (index % 4) / 10,
                "artifact_work_rate_per_100_ticks": 0.5 + (index % 3),
                "social_interaction_rate_per_100_ticks": 0.1 + (index % 5),
                "observation_testing_rate_per_100_ticks": 0.2 + (index % 2),
                "material_processing_rate_per_100_ticks": 0.3 + (index % 3),
                "construction_control_rate_per_100_ticks": 0.4 + (index % 4),
                "culture_coordination_rate_per_100_ticks": 0.5 + (index % 5),
            }
        )

    embedded, profiles, fractions, model = _build_behavior_embedding(rows)

    assert len(embedded) == 24
    assert profiles
    assert fractions
    assert model["status"] == "complete"
    assert model["nonlinear_method"] == "PCA fallback"
    assert model["balanced_agents_per_episode"] == 6
    assert model["n_fit_agents"] == 24
    assert "condition" not in model["features"]
    assert "population_size" not in model["features"]
    assert all("behavior_cluster" in row for row in embedded)


def test_four_condition_regime_embedding_uses_common_seed_endpoints_only():
    episodes = []
    conditions = (
        "full",
        "no-explicit-culture",
        "no-communication",
        "independent-search",
    )
    for condition_index, condition in enumerate(conditions):
        for seed in (1, 2):
            scale = 1.0 + condition_index + seed / 10
            episodes.append(
                {
                    "condition": condition,
                    "population_size": 50 * seed,
                    "seed": seed,
                    "mean_distance_traveled": 20.0 * scale,
                    "mean_distinct_cells_visited": 12.0 * scale,
                    "mean_net_displacement": 5.0 * scale,
                    "displacement_efficiency": 0.2 + 0.02 * scale,
                    "exploration_efficiency": 0.3 + 0.01 * scale,
                    "successful_moves_per_agent": 10.0 * scale,
                    "early_mobility_rate": 0.04 * scale,
                    "late_mobility_rate": 0.02 * scale,
                    "moving_fraction_auc": 0.5 + 0.02 * scale,
                }
            )

    embedded, profiles, model = _build_regime_embedding(episodes)

    assert len(embedded) == 8
    assert profiles
    assert model["status"] == "complete"
    assert model["n_episodes"] == 8
    assert "condition" not in model["features"]
    assert all("regime_pc1" in row and "regime_pc2" in row for row in embedded)


def test_self_organization_renderer_skips_shared_conditions_absent_from_study(tmp_path):
    episodes = [
        {"condition": condition, "population_size": 100, "seed": 1}
        for condition in ("full", "no-explicit-culture")
    ]
    metrics = {
        name: {"mean": value, "ci95_low": value - 0.01, "ci95_high": value + 0.01}
        for name, value in (
            ("path_length_gini", 0.2),
            ("final_spatial_entropy", 0.7),
            ("final_artifact_attraction", 0.4),
            ("mean_regional_switch_fraction", 0.1),
        )
    }
    estimates = {
        condition: {"100": {"metrics": metrics}}
        for condition in ("full", "no-explicit-culture")
    }

    outputs = _render_self_organization(episodes, estimates, tmp_path)

    assert set(outputs) == {"png", "pdf", "svg"}
    assert all(Path(path).is_file() for path in outputs.values())
