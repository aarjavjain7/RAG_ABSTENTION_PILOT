"""Recreate the pooled held-out risk-coverage figure from saved pilot scores."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORES_FILE = ROOT / "data" / "pilot_scores.csv"
MANIFEST_FILE = ROOT / "data" / "run_manifest.json"
OUT_FILE = ROOT / "figures" / "risk_coverage.png"


def main() -> None:
    df = pd.read_csv(SCORES_FILE)
    df = df.loc[df["split"].eq("test")].copy()

    sufficient = df["structurally_sufficient"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if sufficient.isna().any():
        raise ValueError("Unexpected values in structurally_sufficient column")
    df["insufficient"] = ~sufficient
    df["score"] = pd.to_numeric(df["answerability_score"], errors="raise")

    if len(df) != 480 or df["question_id"].nunique() != 80:
        raise ValueError("Expected 480 held-out contexts from 80 questions")
    if df["condition"].nunique() != 6:
        raise ValueError("Expected all six evidence conditions")

    with MANIFEST_FILE.open(encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    ordered = df.sort_values("score", ascending=False).reset_index(drop=True)
    accepted_count = np.arange(1, len(ordered) + 1)
    coverage = 100 * accepted_count / len(ordered)
    risk = 100 * ordered["insufficient"].astype(int).cumsum() / accepted_count

    accepted = df.loc[df["score"] >= threshold]
    n_accepted = len(accepted)
    n_insufficient = int(accepted["insufficient"].sum())
    cutoff_coverage = 100 * n_accepted / len(df)
    cutoff_risk = 100 * n_insufficient / n_accepted

    if (n_accepted, n_insufficient) != (111, 32):
        raise ValueError(
            "Uploaded scores do not match this figure's expected cutoff counts: "
            f"accepted={n_accepted}, structurally_insufficient={n_insufficient}"
        )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(coverage, risk, color="#2166ac", linewidth=1.8)
    ax.plot(cutoff_coverage, cutoff_risk, "o", color="#b03a2e", markersize=8, zorder=5)
    ax.annotate(
        f"Frozen cutoff t={threshold:.4f}\n"
        f"{cutoff_coverage:.1f}% coverage; {cutoff_risk:.1f}% risk",
        xy=(cutoff_coverage, cutoff_risk),
        xytext=(45, 72),
        fontsize=11,
        color="#b03a2e",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#b03a2e"),
        arrowprops=dict(arrowstyle="->", color="#b03a2e"),
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Coverage (% of held-out contexts accepted)", fontsize=12)
    ax.set_ylabel("False-accept risk (%)", fontsize=12)
    ax.set_title("Risk–coverage curve", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.text(
        0.01,
        -0.02,
        "Pooled held-out set: 80 source questions × six conditions; support-proxy labels.",
        fontsize=9,
        color="#555",
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_FILE}")
    print(
        f"accepted={n_accepted}; insufficient={n_insufficient}; "
        f"coverage={cutoff_coverage:.1f}%; risk={cutoff_risk:.1f}%"
    )


if __name__ == "__main__":
    main()
