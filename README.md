# HotpotQA Abstention Pilot

This repository contains the code and saved artifacts for a small HotpotQA pilot. The experiment asks whether an answerability cutoff selected on clean evidence behaves similarly when evidence is removed, repeated, or combined with a topical distractor.

The pilot uses `google/flan-t5-base` to score whether a question-context pair appears answerable. It does not generate answers, evaluate answer correctness, or measure hallucination reduction. Labels for edited contexts are structural proxies based on HotpotQA supporting-fact annotations.

## Repository contents

- `experiment/colab_pilot.py`: full source for the HotpotQA scoring run.
- `experiment/RAG_Abstention_Feasibility_Pilot.ipynb`: Colab-ready notebook for the same pilot.
- `experiment/requirements-colab.txt`: dependencies used by the notebook.
- `data/pilot_scores.csv`: saved scores and context metadata for 120 questions across six conditions (720 rows).
- `data/human_audit_reviewed.csv`: the single-reviewer audit labels for 70 contexts.
- `data/run_manifest.json`: saved run settings and environment details.
- `analysis/analyze_family_calibration.py`: post-hoc family-matched cutoff analysis using saved scores only.
- `analysis/plot_risk_coverage.py`: reproduces the pooled held-out risk-coverage plot.
- `results/`: saved condition and family-analysis summaries.
- `figures/risk_coverage.png`: the reproduced risk-coverage figure.

## Recreate the figure

The graph uses the held-out portion of `data/pilot_scores.csv` and the threshold in `data/run_manifest.json`. It does not require downloading the dataset or running the model.

```bash
python -m pip install -r requirements-plot.txt
python analysis/plot_risk_coverage.py
```

The script saves `figures/risk_coverage.png`. It checks that the held-out data contains 480 contexts from 80 questions, and that the frozen cutoff accepts 111 contexts, including 32 structurally insufficient contexts. This corresponds to 23.1% coverage and 28.8% structural false-accept risk.

## Re-run the model experiment

Open `experiment/RAG_Abstention_Feasibility_Pilot.ipynb` in Google Colab and run the notebook cells. A GPU can make inference faster; it is not needed to regenerate the graph from the saved CSV.

The original run requested the HotpotQA dataset revision `main` and did not record an immutable model revision. A future rerun may therefore use different data or model weights and need not reproduce the saved scores exactly. The saved CSV and manifest are the records behind the reported pilot results.

## Re-run the post-hoc family analysis

```bash
python -m pip install pandas numpy
python analysis/analyze_family_calibration.py \
  --input data/pilot_scores.csv \
  --outdir results/family_analysis
```

This analysis uses the saved score file and does not run model inference. It is exploratory: neither noisy-family calibration pool met the empirical 10% target, and the thresholds do not provide a formal risk guarantee.

## Data attribution and limitations

HotpotQA is distributed under the [Creative Commons Attribution-ShareAlike 4.0 license](https://huggingface.co/datasets/hotpotqa/hotpot_qa). The CSV includes question and evidence text derived from the dataset; retain the dataset attribution and license when redistributing it. See `DATA_AND_CODE_LICENSES.md` for the project-specific note.

The model score is a YES-versus-NO sequence-likelihood preference, not a calibrated probability. The human audit contains one reviewer's judgments and is not an adjudicated gold standard. The pilot evaluates an answerability gate only; it does not establish answer correctness, groundedness, or a reduction in hallucinations.
