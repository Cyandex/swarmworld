import json

import numpy as np
import pytest

from biofoundry import analysis as study_analysis


def test_plotting_errors_clip_floating_point_ci_boundary_noise() -> None:
    errors = study_analysis._nonnegative_yerr(
        np.asarray([0.0, 0.5]),
        np.asarray([np.nextafter(0.0, 1.0), 0.4]),
        np.asarray([0.2, 0.6]),
    )

    np.testing.assert_allclose(errors, [[0.0, 0.1], [0.2, 0.1]], atol=1e-15)


def test_analysis_bootstrap_stream_is_stable_by_statistic_key() -> None:
    values = [0.1, 0.2, 0.5, 0.9]
    first = study_analysis._bootstrap_mean(
        values, study_analysis._analysis_rng("gain", "full", 100), 200
    )
    _ = study_analysis._bootstrap_mean(
        values, study_analysis._analysis_rng("unrelated"), 200
    )
    repeated = study_analysis._bootstrap_mean(
        values, study_analysis._analysis_rng("gain", "full", 100), 200
    )
    assert repeated == first


def _episode(condition: str, seed: int, composition_synergy: float | None) -> dict[str, object]:
    return {
        "condition": condition,
        "seed": seed,
        "engine_revision": 9,
        "position_summary_version": 2,
        "initial_positions": [[3, 4]],
        "macroturn_phases": [0],
        "model_calls": 1,
        "decision_opportunities": 1,
        "model_usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "action_counts": {"MOVE": 1, "WAIT": 2},
        "action_budget": {"used": 1, "blocked": 0},
        "model_errors": 0,
        "research": {
            "outcome_success": False,
            "emergent_success": False,
            "best_artifact_performance": 0.1,
            "best_current_program_peak_performance": 0.1,
            "best_final_state_artifact_performance": 0.1,
            "portfolio_resilience": 0.05,
            "service_breadth": 1,
            "self_organized_portfolio": False,
            "best_behavioral_novelty": 0.1,
            "best_material_utility": 0.4,
            "composition_synergy": composition_synergy,
            "counts": {"validated_inventions": 0},
        },
        "swarm_metrics": {
            "excess_normalized_specialization": 0.0,
            "causal_composition_success": False,
        },
    }


def test_analysis_recovers_legacy_initial_positions_from_tick_zero_trace(
    tmp_path, monkeypatch
) -> None:
    records = []
    for condition, mislabeled_final_position in (
        ("full", [8, 9]),
        ("no-communication", [10, 11]),
    ):
        trace = tmp_path / f"{condition}.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "type": "snapshot",
                    "snapshot": {
                        "tick": 0,
                        "agents": {"x": [3], "y": [4]},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        record = _episode(condition, 17, None)
        record.pop("position_summary_version")
        record["initial_positions"] = [mislabeled_final_position]
        record["output"] = str(trace)
        records.append(record)
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(study_analysis, "_plot_study", lambda *_args: None)

    result = study_analysis.analyze_study(source, tmp_path / "analysis", resamples=10)

    audit = result["compute_audit"]["decision_opportunities"][0]
    assert audit["initial_positions_verified"] is True
    assert audit["initial_positions_matched"] is True
    assert audit["initial_position_sources"] == "trace-tick-0"
    assert audit["valid"] is True


def test_analysis_recovers_corrected_program_fork_metric_from_trace(
    tmp_path, monkeypatch
) -> None:
    trace = tmp_path / "legacy.jsonl"
    trace.write_text(
        json.dumps(
            {
                "type": "snapshot",
                "snapshot": {
                    "tick": 1,
                    "agents": {
                        "ids": ["agent_000000", "agent_000001"],
                        "x": [3, 4],
                        "y": [4, 4],
                    },
                    "program_catalog": [
                        {"program_id": "starter", "authors": ["system"]},
                        {"program_id": "parent", "authors": ["agent_000000"]},
                        {"program_id": "own-child", "authors": ["agent_000000"]},
                        {"program_id": "other-child", "authors": ["agent_000001"]},
                    ],
                    "program_lineage": [
                        {
                            "parent_program_id": "starter",
                            "child_program_id": "parent",
                            "author": "agent_000000",
                        },
                        {
                            "parent_program_id": "parent",
                            "child_program_id": "own-child",
                            "author": "agent_000000",
                        },
                        {
                            "parent_program_id": "parent",
                            "child_program_id": "other-child",
                            "author": "agent_000001",
                        },
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = _episode("full", 17, None)
    record["output"] = str(trace)
    record["technology_ecology"] = {
        "program_forks": 3,
        "cross_agent_program_forks": 2,
        "cross_agent_program_fork_fraction": 2 / 3,
    }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps([record]), encoding="utf-8")
    monkeypatch.setattr(study_analysis, "_plot_study", lambda *_args: None)

    result = study_analysis.analyze_study(source, tmp_path / "analysis", resamples=10)

    endpoint = result["conditions"]["full"]["endpoints"][
        "cross_agent_program_fork_fraction"
    ]
    assert endpoint["mean"] == 0.5
    assert result["program_fork_metric_sources"] == {"trace-recovered-v2": 1}


def test_analysis_reports_unobserved_composition_as_missing(tmp_path, monkeypatch) -> None:
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            [
                _episode("full", 17, None),
                _episode("no-communication", 17, None),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(study_analysis, "_plot_study", lambda *_args: None)

    result = study_analysis.analyze_study(source, tmp_path / "analysis", resamples=10)
    composition = result["conditions"]["full"]["endpoints"]["composition_synergy"]
    assert composition == {
        "mean": None,
        "ci95_low": None,
        "ci95_high": None,
        "n_observed": 0,
    }
    paired = result["paired_differences_from_full"]["no-communication"]["endpoints"]
    assert paired["composition_synergy"]["full_minus_ablation"] is None
    assert paired["composition_synergy"]["n_observed_pairs"] == 0


def test_analysis_combines_separate_studies_with_explicit_labels(
    tmp_path, monkeypatch
) -> None:
    full = tmp_path / "full.json"
    single = tmp_path / "single.json"
    full.write_text(json.dumps([_episode("full", 17, 0.02)]), encoding="utf-8")
    single.write_text(json.dumps([_episode("full", 17, 0.01)]), encoding="utf-8")
    monkeypatch.setattr(study_analysis, "_plot_study", lambda *_args: None)

    result = study_analysis.analyze_study(
        [full, single],
        tmp_path / "analysis",
        resamples=10,
        source_labels=["full", "single-agent"],
    )

    assert result["conditions"]["full"]["n"] == 1
    assert result["conditions"]["single-agent"]["n"] == 1
    paired = result["paired_differences_from_full"]["single-agent"]
    assert paired["n_pairs"] == 1
    assert paired["endpoints"]["composition_synergy"]["full_minus_ablation"] == 0.01


def test_analysis_reports_population_scaling_and_independent_search_gain(
    tmp_path, monkeypatch
) -> None:
    records = []
    for population, full_value, isolated_value in ((4, 0.4, 0.3), (16, 0.7, 0.5)):
        for condition, value in (
            ("full", full_value),
            ("independent-search", isolated_value),
        ):
            record = _episode(condition, 3001, 0.0)
            record["agents"] = population
            record["population_size"] = population
            record["research"]["best_artifact_performance"] = value
            record["discovery_frontier"] = {"normalized_auc": value}
            record["ecosystem_assay"] = {"intact": {"resilience_auc": value}}
            records.append(record)
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(study_analysis, "_plot_study", lambda *_args: None)
    monkeypatch.setattr(
        study_analysis, "_plot_population_scaling", lambda *_args: None
    )

    result = study_analysis.analyze_study(source, tmp_path / "analysis", resamples=10)

    assert result["population_scaling"]["full"]["16"]["n"] == 1
    gain = result["cultural_gain"]["full"]["16"]["endpoints"]
    assert gain["artifact_performance"][
        "collective_minus_independent_envelope"
    ] == pytest.approx(0.2)
    assert (tmp_path / "analysis" / "population-scaling.csv").exists()
    assert (tmp_path / "analysis" / "cultural-gain.csv").exists()
    assert (tmp_path / "analysis" / "pairwise-population-contrasts.csv").exists()
    contrast = result["pairwise_population_contrasts"][
        "full-minus-independent-search"
    ]["populations"]["16"]["endpoints"]["artifact_performance"]
    assert contrast["condition_a_minus_b"] == pytest.approx(0.2)
    assert contrast["n_pairs"] == 1
    assert (tmp_path / "analysis" / "compute-audit.csv").exists()
    audit = result["compute_audit"]["decision_opportunities"][0]
    assert audit["decision_opportunities_matched"] is True
    assert audit["initial_positions_matched"] is True
    assert audit["macroturn_phases_matched"] is True
    assert audit["valid"] is True


def test_analysis_reports_paired_held_out_checkpoint_trajectories(
    tmp_path, monkeypatch
) -> None:
    records = []
    values = {
        ("full", 3301): (0.10, 0.30),
        ("full", 3302): (0.20, 0.40),
        ("independent-search", 3301): (0.15, 0.25),
        ("independent-search", 3302): (0.10, 0.35),
    }
    for (condition, seed), checkpoint_values in values.items():
        record = _episode(condition, seed, 0.0)
        record["agents"] = 100
        record["population_size"] = 100
        record["held_out_checkpoints"] = [
            {
                "tick": tick,
                "model_calls": tick,
                "model_usage": {"total_tokens": tick * 10},
                "held_out_generalization": {
                    "resilience_auc": {"mean": value}
                },
            }
            for tick, value in zip((400, 800), checkpoint_values, strict=True)
        ]
        records.append(record)
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(study_analysis, "_plot_study", lambda *_args: None)
    monkeypatch.setattr(
        study_analysis, "_plot_population_scaling", lambda *_args: None
    )
    monkeypatch.setattr(
        study_analysis, "_plot_held_out_checkpoints", lambda *_args: None
    )

    result = study_analysis.analyze_study(
        source, tmp_path / "analysis", resamples=100
    )

    trajectory = result["held_out_checkpoint_trajectories"]["100"]
    assert trajectory["conditions"]["full"]["400"]["mean"] == pytest.approx(
        0.15
    )
    gain = trajectory["paired_differences_from_full"]["independent-search"]
    assert gain["400"]["full_minus_condition"] == pytest.approx(0.025)
    assert gain["800"]["full_minus_condition"] == pytest.approx(0.05)
    assert (tmp_path / "analysis" / "held-out-checkpoint-seed-values.csv").exists()
    assert (tmp_path / "analysis" / "held-out-checkpoint-estimates.csv").exists()
    assert (
        tmp_path / "analysis" / "held-out-checkpoint-paired-gains.csv"
    ).exists()
