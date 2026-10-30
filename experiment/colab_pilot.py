# %% [markdown]
# Reliable RAG abstention: controlled feasibility pilot
#
# This pilot tests one narrow question: does a fixed answerability threshold
# transfer from clean evidence to repeated and distracting context, and does
# it respond when an omitted annotated support sentence is restored?
#
# It scores an evidence sufficiency decision only. It does not generate
# answers, measure answer factuality, or establish real-world reliability.

# %%
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import datasets
import transformers
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# %% [markdown]
# ## Frozen pilot settings
#
# Keep these settings fixed before inspecting test results. This small pilot
# is for feasibility and failure discovery, not a final power calculation.

# %%
SEED = 42
N_QUESTIONS = 120
N_CALIBRATION = 40
MODEL_NAME = "google/flan-t5-base"
RISK_TARGET = 0.10
MIN_ACCEPTED_CALIBRATION = 5
BATCH_SIZE = 8
MAX_INPUT_TOKENS = 512
DATASET_REPO = "hotpotqa/hotpot_qa"
DATASET_REVISION = "main"  # Record the resolved fingerprint in the run manifest.
OUTPUT_DIR = Path("rag_abstention_pilot_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# %% [markdown]
# ## Load HotpotQA and build matched evidence contexts
#
# The labels below mean that all selected HotpotQA supporting-fact sentence
# identifiers are present. That is an auditable structural proxy for support,
# not a guarantee that no alternate evidence exists or that the source is true.
# The exported audit sample must be checked by people before scientific claims.

# %%
dataset = load_dataset(DATASET_REPO, "distractor", split="validation",
                      revision=DATASET_REVISION)
stop_words = {
    "the", "a", "an", "is", "was", "were", "are", "in", "of", "to", "for",
    "and", "or", "on", "at", "by", "from", "with", "what", "who", "where",
    "when", "which", "how", "did", "does", "do", "name", "tell", "me",
}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop_words}


def get_example(example, source_index: int):
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    support = example["supporting_facts"]
    support_pairs = list(zip(support["title"], support["sent_id"]))
    if len(support_pairs) < 2:
        return None
    title_to_sentences = {title: sents for title, sents in zip(titles, sentences)}
    if any(title not in title_to_sentences or sid >= len(title_to_sentences[title])
           for title, sid in support_pairs):
        return None

    # Keep the annotated evidence sentences as short, auditable passages.
    support_pairs = list(dict.fromkeys(support_pairs))
    support_passages = [
        (title, sid, title_to_sentences[title][sid]) for title, sid in support_pairs
    ]
    if len(support_passages) < 2:
        return None

    question = example["question"]
    supporting_titles = set(support["title"])
    query_words = words(question)
    distractor_candidates = []
    for title, sents in zip(titles, sentences):
        if title in supporting_titles or not sents:
            continue
        # Select the strongest lexical-overlap non-support paragraph so the
        # distractor is plausibly topical. Audit it because it may still
        # contain alternate answer evidence.
        text = " ".join(sents[:3])
        overlap = len(query_words & words(title + " " + text))
        distractor_candidates.append((overlap, title, text))
    if not distractor_candidates:
        return None
    _, distractor_title, distractor_text = sorted(
        distractor_candidates, key=lambda x: (-x[0], x[1])
    )[0]

    # Remove the last annotated support sentence deterministically. The
    # audit sample checks whether this really makes the question unanswerable.
    removed = support_passages[-1]
    full_text = "\n".join(f"[{t}] {s}" for t, _, s in support_passages)
    missing_passages = support_passages[:-1]
    missing_text = "\n".join(f"[{t}] {s}" for t, _, s in missing_passages)
    repeated_text = full_text + f"\n[Repeated passage] {support_passages[0][2]}"
    distractor = f"[{distractor_title}] {distractor_text}"

    contexts = {
        "sufficient_clean": (full_text, True),
        "insufficient_clean": (missing_text, False),
        "sufficient_repeat": (repeated_text, True),
        "insufficient_repeat": (missing_text + f"\n[Repeated passage] {support_passages[0][2]}", False),
        "insufficient_distractor": (missing_text + "\n" + distractor, False),
        # This is the missing+distractor context after restoring the omitted
        # support sentence; it is exactly matched to insufficient_distractor.
        "restored_distractor": (full_text + "\n" + distractor, True),
    }
    return {
        "source_index": source_index,
        "question": question,
        "answer": example["answer"],
        "supporting_facts": json.dumps(support_pairs, ensure_ascii=False),
        "removed_fact": json.dumps(removed, ensure_ascii=False),
        "distractor_title": distractor_title,
        "distractor_text": distractor_text,
        "contexts": contexts,
    }


eligible = []
for idx, example in enumerate(dataset):
    item = get_example(example, idx)
    if item is not None:
        eligible.append(item)

if len(eligible) < N_QUESTIONS:
    raise RuntimeError(f"Only {len(eligible)} eligible items; reduce N_QUESTIONS.")
rng = random.Random(SEED)
selected = rng.sample(eligible, N_QUESTIONS)
rng.shuffle(selected)
for split, items in (("calibration", selected[:N_CALIBRATION]),
                     ("test", selected[N_CALIBRATION:])):
    for item in items:
        item["split"] = split

rows = []
for item in selected:
    for condition, (context, sufficient) in item["contexts"].items():
        rows.append({
            "question_id": str(item["source_index"]),
            "split": item["split"],
            "condition": condition,
            "question": item["question"],
            "context": context,
            "gold_answer": item["answer"],
            "structurally_sufficient": sufficient,
            "supporting_facts": item["supporting_facts"],
            "removed_fact": item["removed_fact"],
            "distractor_title": item["distractor_title"],
            "distractor_text": item["distractor_text"],
        })
contexts_df = pd.DataFrame(rows)
print(f"Questions: {len(selected)} ({N_CALIBRATION} calibration, "
      f"{N_QUESTIONS-N_CALIBRATION} test); context rows: {len(contexts_df)}")
print(contexts_df.groupby(["split", "condition"]).size().unstack(fill_value=0))


# %% [markdown]
# ## Create a blinded human audit file
#
# Review every row in this sample without looking at the model score. Mark
# `human_sufficient` as 1 if the shown context really supports the requested
# answer under a context-only policy, 0 if it does not, or leave blank if
# uncertain. Mark whether the distractor accidentally supplies support.

# %%
audit = contexts_df.sample(n=min(70, len(contexts_df)), random_state=SEED).copy()
audit = audit[["question_id", "condition", "question", "context", "gold_answer",
               "supporting_facts", "removed_fact", "distractor_title"]]
audit["human_sufficient"] = ""
audit["distractor_adds_support"] = ""
audit["notes"] = ""
audit_path = OUTPUT_DIR / "human_audit.csv"
audit.to_csv(audit_path, index=False)
print(f"Audit file saved to {audit_path.resolve()}")


# %% [markdown]
# ## Score answerability with a fixed open model
#
# Flan-T5 is asked whether the supplied context has enough evidence. We compare
# its sequence log-likelihood for `YES` and `NO`, then normalize the two scores
# into an answerability ranking score. This is a useful pilot signal, not a
# calibrated probability. No answer is generated in this pilot.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


def score_yes_probability(texts: list[str]) -> np.ndarray:
    all_scores = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=MAX_INPUT_TOKENS, return_tensors="pt")
        batch_n = len(batch_texts)
        yes = tokenizer(text_target=["YES"] * batch_n, padding=True,
                         return_tensors="pt")["input_ids"]
        no = tokenizer(text_target=["NO"] * batch_n, padding=True,
                       return_tensors="pt")["input_ids"]
        labels = torch.cat([yes, no], dim=0)
        labels[labels == tokenizer.pad_token_id] = -100
        input_ids = torch.cat([enc["input_ids"], enc["input_ids"]], dim=0).to(device)
        attention_mask = torch.cat([enc["attention_mask"], enc["attention_mask"]], dim=0).to(device)
        labels = labels.to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels).logits
            token_log_probs = torch.log_softmax(logits, dim=-1)
            safe_labels = labels.masked_fill(labels == -100, 0)
            selected_log_probs = token_log_probs.gather(
                -1, safe_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (labels != -100).float()
            sequence_scores = (selected_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)
            pair_scores = sequence_scores.reshape(2, batch_n).transpose(0, 1)
            yes_probability = torch.softmax(pair_scores, dim=-1)[:, 0]
        all_scores.extend(yes_probability.cpu().numpy().tolist())
    return np.asarray(all_scores, dtype=float)


def make_prompt(row) -> str:
    return (
        "Decide whether the evidence below contains enough information to answer "
        "the question using only these passages. Answer YES only if the requested "
        "answer is supported; otherwise answer NO.\n\n"
        f"Question: {row.question}\n\nEvidence:\n{row.context}\n\nLabel:"
    )


contexts_df["prompt"] = contexts_df.apply(make_prompt, axis=1)
contexts_df["answerability_score"] = score_yes_probability(contexts_df["prompt"].tolist())
contexts_df.drop(columns=["prompt"], inplace=True)
score_path = OUTPUT_DIR / "pilot_scores.csv"
contexts_df.to_csv(score_path, index=False)
print(f"Scores saved to {score_path.resolve()}")


# %% [markdown]
# ## Select the threshold on clean calibration contexts only
#
# Threshold selection uses only calibration questions and the two clean
# conditions. The threshold is then frozen for every test condition.

# %%
cal = contexts_df[(contexts_df.split == "calibration") &
                  (contexts_df.condition.isin(["sufficient_clean", "insufficient_clean"]))].copy()


def false_accept_risk(frame: pd.DataFrame, threshold: float) -> tuple[float, int, int]:
    accepted = frame.answerability_score >= threshold
    n_accepted = int(accepted.sum())
    n_unsupported = int((accepted & ~frame.structurally_sufficient).sum())
    risk = n_unsupported / n_accepted if n_accepted else float("nan")
    return risk, n_accepted, n_unsupported


candidates = sorted(set([0.0, 1.0] + cal.answerability_score.tolist()))
feasible = []
for threshold in candidates:
    risk, n_accepted, n_unsupported = false_accept_risk(cal, threshold)
    if n_accepted >= MIN_ACCEPTED_CALIBRATION and risk <= RISK_TARGET:
        feasible.append((n_accepted, -threshold, threshold, risk, n_unsupported))

if feasible:
    _, _, threshold, calibration_risk, calibration_unsupported = max(feasible)
    threshold_status = "target_met"
else:
    # Prespecified fallback: choose the threshold with lowest empirical risk,
    # then highest coverage. Mark the target infeasible in every report.
    fallback = []
    for t in candidates:
        risk, n_accepted, n_unsupported = false_accept_risk(cal, t)
        if n_accepted >= MIN_ACCEPTED_CALIBRATION:
            fallback.append((risk, -n_accepted, t, n_unsupported))
    if not fallback:
        raise RuntimeError("No threshold accepts the minimum calibration count.")
    calibration_risk, _, threshold, calibration_unsupported = min(fallback)
    threshold_status = "risk_target_infeasible"

print({"threshold": float(threshold), "target": RISK_TARGET,
       "calibration_risk": float(calibration_risk),
       "calibration_unsupported": int(calibration_unsupported),
       "status": threshold_status})

contexts_df["answered_by_gate"] = contexts_df.answerability_score >= threshold
contexts_df["threshold"] = float(threshold)
contexts_df["threshold_status"] = threshold_status


# %% [markdown]
# ## Held-out results and paired diagnostics
#
# False-acceptance risk here means an accepted context is missing at least one
# benchmark-annotated support sentence. It is not the same as unsupported
# generated-answer risk. This pilot makes no claim about final answer quality.

# %%
test = contexts_df[contexts_df.split == "test"].copy()
summary_rows = []
for condition, frame in test.groupby("condition", sort=False):
    accepted = frame.answered_by_gate
    n = len(frame)
    n_answered = int(accepted.sum())
    n_unsupported = int((accepted & ~frame.structurally_sufficient).sum())
    n_sufficient = int(frame.structurally_sufficient.sum())
    n_sufficient_refused = int((~accepted & frame.structurally_sufficient).sum())
    summary_rows.append({
        "condition": condition,
        "n_questions": n,
        "n_answered": n_answered,
        "coverage": n_answered / n if n else np.nan,
        "structural_false_accept_risk": n_unsupported / n_answered if n_answered else np.nan,
        "false_accept_rate_on_insufficient": n_unsupported / int((~frame.structurally_sufficient).sum())
            if (~frame.structurally_sufficient).sum() else np.nan,
        "unnecessary_abstention_rate_on_sufficient": n_sufficient_refused / n_sufficient
            if n_sufficient else np.nan,
        "mean_score": frame.answerability_score.mean(),
    })

condition_order = ["sufficient_clean", "insufficient_clean", "sufficient_repeat",
                   "restored_distractor", "insufficient_repeat", "insufficient_distractor"]
wide = test.pivot(index="question_id", columns="condition", values="answered_by_gate")
wide = wide[condition_order]
answered_matrix = wide.to_numpy(dtype=bool)
sufficient_matrix = test.pivot(index="question_id", columns="condition",
                               values="structurally_sufficient")[condition_order].to_numpy(dtype=bool)
bootstrap_rng = np.random.default_rng(SEED)
bootstrap = {condition: {"coverage": [], "false_accept_risk": [],
                         "unnecessary_abstention": []}
             for condition in condition_order}
paired_bootstrap = {"flip_repeat": [], "flip_distractor": [], "restoration_recovery": []}
for _ in range(2000):
    sample_idx = bootstrap_rng.integers(0, len(wide), size=len(wide))
    decisions = answered_matrix[sample_idx]
    sufficient = sufficient_matrix[sample_idx]
    for j, condition in enumerate(condition_order):
        accepted = decisions[:, j]
        enough = sufficient[:, j]
        bootstrap[condition]["coverage"].append(float(accepted.mean()))
        bootstrap[condition]["false_accept_risk"].append(
            float((accepted & ~enough).sum() / accepted.sum()) if accepted.sum() else np.nan
        )
        bootstrap[condition]["unnecessary_abstention"].append(
            float((~accepted & enough).sum() / enough.sum()) if enough.sum() else np.nan
        )
    paired_bootstrap["flip_repeat"].append(float((decisions[:, 0] != decisions[:, 2]).mean()))
    paired_bootstrap["flip_distractor"].append(float((decisions[:, 0] != decisions[:, 3]).mean()))
    paired_bootstrap["restoration_recovery"].append(
        float(((~decisions[:, 5]) & decisions[:, 3]).sum() / (~decisions[:, 5]).sum())
        if (~decisions[:, 5]).sum() else np.nan
    )


def percentile_interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return [float(x) for x in np.quantile(finite, [0.025, 0.975])] if len(finite) else [None, None]


for row in summary_rows:
    condition = row["condition"]
    for metric, key in (("coverage", "coverage"),
                        ("structural_false_accept_risk", "false_accept_risk"),
                        ("unnecessary_abstention_rate_on_sufficient", "unnecessary_abstention")):
        row[metric + "_95ci"] = percentile_interval(bootstrap[condition][key])

summary = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "condition_summary.csv"
summary.to_csv(summary_path, index=False)
print(summary.to_string(index=False))

paired = {
    "n_test_questions": int(wide.shape[0]),
    "flip_sufficient_clean_vs_repeat": float((answered_matrix[:, 0] != answered_matrix[:, 2]).mean()),
    "flip_sufficient_clean_vs_distractor": float((answered_matrix[:, 0] != answered_matrix[:, 3]).mean()),
    "restoration_recovery_with_distractor": float(
        ((~answered_matrix[:, 5]) & answered_matrix[:, 3]).sum() /
        (~answered_matrix[:, 5]).sum()
        if (~answered_matrix[:, 5]).sum() else np.nan
    ),
    "flip_sufficient_clean_vs_repeat_95ci": percentile_interval(paired_bootstrap["flip_repeat"]),
    "flip_sufficient_clean_vs_distractor_95ci": percentile_interval(paired_bootstrap["flip_distractor"]),
    "restoration_recovery_with_distractor_95ci": percentile_interval(
        paired_bootstrap["restoration_recovery"]
    ),
}
print("Paired diagnostics:", paired)


# %% [markdown]
# ## Plot the answerability score shift
#
# This plot is descriptive. The calibration threshold is shown so the held-out
# changes are visible without selecting a new operating point for each noise.

# %%
import matplotlib.pyplot as plt

plot_order = ["sufficient_clean", "insufficient_clean", "sufficient_repeat",
              "restored_distractor", "insufficient_repeat", "insufficient_distractor"]
fig, ax = plt.subplots(figsize=(12, 5))
scores_by_condition = [test.loc[test.condition == condition, "answerability_score"].to_numpy()
                       for condition in plot_order]
ax.boxplot(scores_by_condition, tick_labels=plot_order, showfliers=False)
ax.tick_params(axis="x", labelrotation=25)
for tick in ax.get_xticklabels():
    tick.set_ha("right")
ax.axhline(float(threshold), color="crimson", linestyle="--", label="Frozen calibration threshold")
ax.set_ylabel("YES-vs-NO ranking score (not calibrated probability)")
ax.set_title("Held-out answerability scores across matched evidence conditions")
fig.suptitle("")
ax.legend()
fig.tight_layout()
plot_path = OUTPUT_DIR / "score_distributions.png"
fig.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.show()
print(f"Plot saved to {plot_path.resolve()}")


# %% [markdown]
# ## Save a run manifest
#
# Preserve this JSON, the CSV files, the human audit, and the exact Colab
# notebook output. Replace no missing values with guessed numbers.

# %%
manifest = {
    "dataset": DATASET_REPO + ":distractor/validation",
    "dataset_revision_requested": DATASET_REVISION,
    "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
    "model": MODEL_NAME,
    "model_revision": getattr(model.config, "_commit_hash", None),
    "library_versions": {
        "datasets": datasets.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
    "seed": SEED,
    "n_questions": N_QUESTIONS,
    "n_calibration": N_CALIBRATION,
    "calibration_conditions": ["sufficient_clean", "insufficient_clean"],
    "risk_target": RISK_TARGET,
    "threshold": float(threshold),
    "threshold_status": threshold_status,
    "score_definition": "softmax of mean sequence log likelihoods for YES and NO",
    "device": device,
    "max_input_tokens": MAX_INPUT_TOKENS,
    "limitations": [
        "pilot only; no trained proposed scorer",
        "support labels are based on annotated support sentence presence",
        "no answer generation or answer-groundedness evaluation",
        "human audit required before scientific claims",
    ],
    "paired_diagnostics": paired,
}
(OUTPUT_DIR / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("Outputs:", OUTPUT_DIR.resolve())
