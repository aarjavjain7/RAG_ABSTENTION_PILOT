"""Exploratory, question-grouped calibration-family analysis for the saved pilot.

This reads existing scores only; it does not run a model. Thresholds are selected
on calibration questions, then frozen and evaluated on disjoint test questions.
The 10% empirical risk target and minimum of five calibration accepts match the
original pilot. The analysis is post-hoc and cannot establish a formal guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FAMILIES = {
    "clean": ("sufficient_clean", "insufficient_clean"),
    "repeat": ("sufficient_repeat", "insufficient_repeat"),
    "distractor": ("restored_distractor", "insufficient_distractor"),
}
RISK_TARGET = 0.10
MIN_ACCEPTED = 5
ORIGINAL_CUTOFF = 0.7407910823822021
BOOTSTRAPS = 10_000
SEED = 20261027


def select_cutoff(frame: pd.DataFrame) -> dict:
    """Choose maximum calibration coverage under the original empirical rule."""
    scores = frame["answerability_score"].to_numpy(dtype=float)
    sufficient = frame["structurally_sufficient"].to_numpy(dtype=bool)
    candidates = np.unique(np.r_[0.0, 1.0, scores])
    feasible = []
    observed = []
    for cutoff in candidates:
        accepted = scores >= cutoff
        count = int(accepted.sum())
        unsafe = int((accepted & ~sufficient).sum())
        risk = unsafe / count if count else float("nan")
        observed.append((float(cutoff), count, unsafe, risk))
        if count >= MIN_ACCEPTED and risk <= RISK_TARGET:
            feasible.append((count, -float(cutoff), float(cutoff), unsafe, risk))
    if feasible:
        count, _, cutoff, unsafe, risk = max(feasible)
        status = "target_met_empirically"
    else:
        eligible = [x for x in observed if x[1] >= MIN_ACCEPTED]
        if not eligible:
            raise ValueError("No threshold meets the minimum accepted calibration count.")
        cutoff, count, unsafe, risk = min(eligible, key=lambda x: (x[3], -x[1], -x[0]))
        status = "risk_target_infeasible"
    return {
        "threshold": float(cutoff),
        "status": status,
        "calibration_n": int(len(frame)),
        "calibration_accepted": int(count),
        "calibration_false_accepts": int(unsafe),
        "calibration_empirical_risk": float(risk),
        "target": RISK_TARGET,
        "minimum_accepted": MIN_ACCEPTED,
    }


def metric_values(frame: pd.DataFrame, cutoff: float) -> dict[str, float]:
    scores = frame["answerability_score"].to_numpy(dtype=float)
    sufficient = frame["structurally_sufficient"].to_numpy(dtype=bool)
    accepted = scores >= cutoff
    n_acc = int(accepted.sum())
    n_suf = int(sufficient.sum())
    n_insuf = int((~sufficient).sum())
    return {
        "n_questions": int(frame["question_id"].nunique()),
        "n_contexts": int(len(frame)),
        "n_accepted": n_acc,
        "coverage": float(n_acc / len(frame)) if len(frame) else float("nan"),
        "structural_false_accept_risk": float((accepted & ~sufficient).sum() / n_acc) if n_acc else float("nan"),
        "false_accept_rate_on_insufficient": float((accepted & ~sufficient).sum() / n_insuf) if n_insuf else float("nan"),
        "unnecessary_abstention_rate_on_sufficient": float((~accepted & sufficient).sum() / n_suf) if n_suf else float("nan"),
    }


def bootstrap_family(frame: pd.DataFrame, cutoff: float, baseline_cutoff: float,
                     rng: np.random.Generator) -> tuple[dict, dict]:
    """Question-level paired bootstrap for family metrics and cutoff deltas."""
    condition_order = list(frame["condition"].drop_duplicates())
    scores_wide = frame.pivot(index="question_id", columns="condition", values="answerability_score")
    labels_wide = frame.pivot(index="question_id", columns="condition", values="structurally_sufficient")
    if set(scores_wide.columns) != set(condition_order) or scores_wide.isna().any().any():
        raise ValueError("Each source question must have exactly one row for both family conditions.")
    scores = scores_wide[condition_order].to_numpy(dtype=float)
    sufficient = labels_wide[condition_order].to_numpy(dtype=bool)
    sample_indices = rng.integers(0, len(scores), size=(BOOTSTRAPS, len(scores)))

    def all_metrics(threshold: float) -> dict[str, np.ndarray]:
        accepted = scores[sample_indices] >= threshold
        enough = sufficient[sample_indices]
        acc = accepted.sum(axis=(1, 2))
        total = accepted.shape[1] * accepted.shape[2]
        n_suff = enough.sum(axis=(1, 2))
        n_insuff = (~enough).sum(axis=(1, 2))
        false_accept = (accepted & ~enough).sum(axis=(1, 2))
        unnecessary = (~accepted & enough).sum(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                "coverage": acc / total,
                "structural_false_accept_risk": np.divide(false_accept, acc, out=np.full(BOOTSTRAPS, np.nan), where=acc > 0),
                "false_accept_rate_on_insufficient": np.divide(false_accept, n_insuff, out=np.full(BOOTSTRAPS, np.nan), where=n_insuff > 0),
                "unnecessary_abstention_rate_on_sufficient": np.divide(unnecessary, n_suff, out=np.full(BOOTSTRAPS, np.nan), where=n_suff > 0),
            }

    adapted = all_metrics(cutoff)
    baseline = all_metrics(baseline_cutoff)
    deltas = {name: adapted[name] - baseline[name] for name in adapted}

    def interval(v: list[float]) -> list[float | None]:
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        return [float(x) for x in np.quantile(arr, [0.025, 0.975])] if len(arr) else [None, None]

    return ({name: interval(v) for name, v in adapted.items()},
            {name: interval(v) for name, v in deltas.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "pilot_scores.csv",
        help="Existing pilot_scores.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "family_analysis",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Scores CSV not found: {args.input}")

    raw = pd.read_csv(args.input)
    required = {"question_id", "split", "condition", "structurally_sufficient", "answerability_score"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    raw["structurally_sufficient"] = raw["structurally_sufficient"].map(
        lambda x: x if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() == "true"
    )
    raw["answerability_score"] = pd.to_numeric(raw["answerability_score"], errors="raise")
    calibration = raw[raw["split"] == "calibration"].copy()
    test = raw[raw["split"] == "test"].copy()
    if set(calibration.question_id) & set(test.question_id):
        raise ValueError("Question leakage: calibration and test question IDs overlap.")

    rng = np.random.default_rng(SEED)
    summary_rows = []
    threshold_records = {}
    for family, (sufficient_condition, insufficient_condition) in FAMILIES.items():
        calibration_family = calibration[calibration.condition.isin(
            [sufficient_condition, insufficient_condition])].copy()
        test_family = test[test.condition.isin([sufficient_condition, insufficient_condition])].copy()
        if len(calibration_family) == 0 or len(test_family) == 0:
            raise ValueError(f"No rows found for family {family}.")
        if set(calibration_family.condition) != {sufficient_condition, insufficient_condition}:
            raise ValueError(f"Calibration family {family} does not include both conditions.")
        if calibration_family.structurally_sufficient.nunique() != 2:
            raise ValueError(f"Calibration family {family} does not contain both label classes.")

        selected = select_cutoff(calibration_family)
        threshold_records[family] = selected
        base = metric_values(test_family, ORIGINAL_CUTOFF)
        tuned = metric_values(test_family, selected["threshold"])
        intervals, delta_intervals = bootstrap_family(
            test_family, selected["threshold"], ORIGINAL_CUTOFF, rng
        )
        baseline_intervals, _ = bootstrap_family(
            test_family, ORIGINAL_CUTOFF, ORIGINAL_CUTOFF, rng
        )
        for policy, cutoff, metric_set, ci_set in (
            ("clean_cutoff_baseline", ORIGINAL_CUTOFF, base, baseline_intervals),
            ("family_calibrated", selected["threshold"], tuned, intervals),
        ):
            row = {"family": family, "policy": policy, "threshold": cutoff, **metric_set}
            for metric, ci in ci_set.items():
                row[f"{metric}_95ci"] = json.dumps(ci)
            if policy == "family_calibrated":
                for metric, ci in delta_intervals.items():
                    row[f"delta_vs_clean_cutoff_{metric}_95ci"] = json.dumps(ci)
            summary_rows.append(row)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary_path = args.outdir / "family_cutoff_test_metrics.csv"
    summary.to_csv(summary_path, index=False)
    result = {
        "analysis": "post-hoc family-matched calibration using existing scores; no model inference",
        "input": str(args.input.resolve()),
        "n_rows": int(len(raw)),
        "n_calibration_questions": int(calibration.question_id.nunique()),
        "n_test_questions": int(test.question_id.nunique()),
        "question_overlap": int(len(set(calibration.question_id) & set(test.question_id))),
        "risk_target": RISK_TARGET,
        "minimum_accepted_calibration": MIN_ACCEPTED,
        "original_clean_cutoff": ORIGINAL_CUTOFF,
        "bootstrap_repetitions": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "analysis_environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "sha256": {
            "source_scores_csv": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "analysis_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "thresholds": threshold_records,
        "interpretation_limits": [
            "exploratory post-hoc analysis, not a preregistered experiment",
            "empirical calibration risk is not a formal risk guarantee",
            "sufficiency is a support-presence proxy and does not measure generated-answer hallucination",
            "same benchmark, one scorer, small question sample, one reviewer audit",
        ],
        "metrics_csv": str(summary_path.resolve()),
    }
    json_path = args.outdir / "family_calibration_manifest.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Selected calibration thresholds:")
    print(json.dumps(threshold_records, indent=2))
    print("Held-out comparison:")
    print(summary.to_string(index=False))
    print("\nSaved family_cutoff_test_metrics.csv and family_calibration_manifest.json in the output directory.")


if __name__ == "__main__":
    main()
