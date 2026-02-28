# LLM Agentic ML Agent

A modular, production-style autonomous data analyst that combines classical ML with an LLM decision brain using OpenRouter, LangChain, and LangGraph.

## Architecture

```
llm_ml_agent/
├── agent/
│   ├── state.py
│   ├── graph.py
│   ├── brain.py
│   └── runner.py
├── tools/
│   ├── inspect.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── feature_importance.py
├── config/
│   └── settings.py
├── main.py
├── requirements.txt
└── README.md
```

## Agentic Workflow

1. `inspect_dataset` loads CSV and summarizes rows, columns, missing %, dtypes.
2. `detect_problem_type` infers regression vs classification.
3. `preprocess` applies leakage-safe preprocessing with split.
4. `train_model` trains baseline models.
5. `evaluate_models` scores models and selects best.
6. `llm_decision_node` decides next step (`train`, `tune`, `try_alternative_model`, `finalize`).
7. Loop continues through LangGraph conditional edges until `finalize` or iteration limit.
8. `feature_importance` extracts top features when tree-based model is selected.

## LLM Decision Strategy

The LLM receives current state context:
- dataset summary
- current metrics
- best model so far
- iteration count
- whether tuning already ran

It returns strict JSON:

```json
{
  "decision": "tune",
  "reason": "Accuracy is below threshold; try hyperparameter tuning"
}
```

The reason is appended to `reasoning_trace` for transparent decision audit.

## OpenRouter Setup

- Base URL: `https://openrouter.ai/api/v1`
- API key is captured securely at runtime if `OPENAI_API_KEY` is not already set.
- Supported model examples:
  - `openai/gpt-4o-mini`
  - `anthropic/claude-3-sonnet`

## Run

```bash
pip install -r requirements.txt
python main.py --file data/train.csv --target Calories --model openai/gpt-4o-mini
```

Short command (Windows):

```powershell
.\run.cmd
```

The launcher auto-detects the target column (prefers `Calories` or `diagnosed_diabetes`, else uses the last CSV column).
If `data/sample_submission.csv` exists, it will prefer that file's prediction column as the target.
If `data/data_description.txt` exists, it is injected as temporary competition-only context into LLM strategy/decision prompts for that run.
Model selection uses frozen 5-fold CV on the training split; holdout is evaluated once at finalization.

With optional overrides:

```powershell
.\run.cmd --max-iterations 1 --show-langgraph
.\run.cmd -Target Calories -MaxIterations 1
.\run.cmd --sample-submission data/sample_submission.csv
.\run.cmd --competition-description data/data_description.txt
.\run.cmd --metric-objective rmsle
```

Optional:

```bash
python main.py --file data/train.csv --target Calories --max-iterations 5
```

## Diagnostics (without extra scripts)

```bash
# Full workflow with graph and detailed report
python main.py --file data/train.csv --test-file data/test.csv --target Calories --max-iterations 1 --show-langgraph

# Save full run log for debugging
python main.py --file data/train.csv --test-file data/test.csv --target Calories --submission-file submissions.csv --max-iterations 1 --show-langgraph > run.log 2>&1
```

## Final Report Includes

- Dataset Summary
- Problem Type
- Models Tried
- Best Model
- Final Metrics
- Hyperparameters
- Feature Importance (top 5 when available)
- Full LLM Reasoning Trace

## Tech Stack

- LangChain + LangGraph
- OpenRouter via ChatOpenAI-compatible client
- Pandas, NumPy, scikit-learn
- Rule-based safety controls (max iterations, no-improvement stop, single tuning pass)

## Workspace Hygiene

- Generated logs such as `run.log`, `run_full.log`, `run_top3.log`, and `debug_run.log` are temporary artifacts.
- Keep only source files under `agent/`, `tools/`, `config/`, plus `main.py`, `requirements.txt`, and `README.md`.
