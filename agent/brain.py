"""LLM-based decision logic for the agent."""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict

from pydantic import BaseModel, Field

from agent.state import AgentGraphState


class DecisionOutput(BaseModel):
    decision: str = Field(description="One of: train, tune, try_alternative_model, finalize")
    reason: str = Field(description="Short explanation for the decision")


class StrategyOutput(BaseModel):
    model_candidates: list[str] = Field(description="Ordered candidate model class names")
    hyperparameter_suggestions: Dict[str, Dict[str, list[Any]]] = Field(
        description="Parameter grid suggestions keyed by model class name"
    )
    reason: str = Field(description="Why these models/hyperparameters were selected")


class PreprocessStrategyOutput(BaseModel):
    numeric_imputer: str = Field(description="One of: mean, median")
    categorical_imputer: str = Field(description="One of: most_frequent, constant")
    categorical_fill_value: str | None = Field(description="Used when categorical_imputer is constant")
    encoder: str = Field(description="One of: onehot, ordinal")
    test_size: float = Field(description="Holdout test ratio between 0.1 and 0.3")
    validation_size_within_train: float = Field(
        description="Validation ratio on train split between 0.1 and 0.3"
    )
    m_estimate_smoothing: float | None = Field(
        default=10.0,
        description="Optional smoothing value for M-estimate target encoder"
    )
    reason: str = Field(description="Why this preprocessing strategy fits the dataset")


class FeatureEngineeringDecisionOutput(BaseModel):
    apply_feature_engineering: bool = Field(
        description="Whether to create synthetic/engineered features"
    )
    reason: str = Field(description="Why you do or do not recommend feature engineering for this dataset")


def _competition_context_for_prompt(state: AgentGraphState, max_chars: int = 4000) -> str:
    context = str(state.get("competition_context", "") or "").strip()
    if not context:
        return ""
    source = str(state.get("competition_context_source", "") or "").strip()
    trimmed = context if len(context) <= max_chars else f"{context[:max_chars].rstrip()}\n...[truncated]"
    source_text = f" source={source}." if source else ""
    return (
        " Temporary competition-only dataset description is provided below; use it as domain context for this run."
        f"{source_text}\n"
        f"{trimmed}\n"
    )


def _normalize_hyperparameter_suggestions(raw: Any) -> Dict[str, Dict[str, list[Any]]]:
    if not isinstance(raw, dict):
        return {}

    normalized: Dict[str, Dict[str, list[Any]]] = {}
    for model_name, params in raw.items():
        key = str(model_name).strip()
        if not key:
            continue

        param_dict: Dict[str, list[Any]] = {}
        if isinstance(params, dict):
            source_dict = params
        elif isinstance(params, list):
            source_dict = {}
            for entry in params:
                if isinstance(entry, dict):
                    source_dict.update(entry)
        else:
            source_dict = {}

        for param_name, values in source_dict.items():
            pkey = str(param_name).strip()
            if not pkey:
                continue
            if isinstance(values, list):
                param_dict[pkey] = values
            else:
                param_dict[pkey] = [values]

        normalized[key] = param_dict

    return normalized


def _quick_feature_engineering_benchmark(state: AgentGraphState) -> tuple[bool, str]:
    """Run a fast sampled A/B check with full preprocessing-compatible feature space.

    Returns (should_apply_feature_engineering, reason).
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, mean_squared_error, mean_squared_log_error
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:
        return True, f"Benchmark skipped (dependency error): {exc}"

    df = state.get("raw_df")
    target_column = state.get("target_column")
    summary = state.get("dataset_summary", {})
    if df is None or target_column not in df.columns:
        return True, "Benchmark skipped (dataset unavailable)"

    working = df.dropna(subset=[target_column]).copy()
    if len(working) < 500:
        return True, "Benchmark skipped (insufficient rows)"

    sample_n = min(50_000, len(working))
    if len(working) > sample_n:
        working = working.sample(n=sample_n, random_state=42)

    identifier_columns = [
        col for col in summary.get("identifier_columns", []) if col in working.columns and col != target_column
    ]
    feature_df = working.drop(columns=[target_column, *identifier_columns], errors="ignore")
    y = working[target_column]

    base_numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    if len(base_numeric_cols) < 2:
        return True, "Benchmark skipped (not enough numeric features)"

    base_X = feature_df.copy()
    if any(col for col in base_numeric_cols):
        base_X[base_numeric_cols] = base_X[base_numeric_cols].apply(pd.to_numeric, errors="coerce")

    target_relations = summary.get("relationship_insights", {}).get("top_target_relations", [])
    relation_features = [str(row.get("feature", "")).strip() for row in target_relations]
    priority = [c for c in relation_features if c in base_numeric_cols]
    if len(priority) < 3:
        variances = (
            base_X[base_numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .var(numeric_only=True)
            .sort_values(ascending=False)
        )
        for col in variances.index.tolist():
            if col not in priority:
                priority.append(col)
            if len(priority) >= 3:
                break

    fe_X = base_X.copy()
    for col in priority[:2]:
        col_num = pd.to_numeric(fe_X[col], errors="coerce")
        fe_X[f"{col}_squared"] = col_num.pow(2)
        fe_X[f"{col}_abs"] = col_num.abs()
    if len(priority) >= 2:
        p0, p1 = priority[0], priority[1]
        p0_num = pd.to_numeric(fe_X[p0], errors="coerce")
        p1_num = pd.to_numeric(fe_X[p1], errors="coerce")
        fe_X[f"{p0}_x_{p1}"] = p0_num * p1_num
        safe_den = p1_num.replace(0, np.nan)
        fe_X[f"{p0}_div_{p1}"] = p0_num / safe_den
        if len(priority) >= 3:
            p2 = priority[2]
            p2_num = pd.to_numeric(fe_X[p2], errors="coerce")
            fe_X[f"{p0}_x_{p2}"] = p0_num * p2_num
            fe_X[f"{p1}_x_{p2}"] = p1_num * p2_num

    numeric_frame = base_X[base_numeric_cols].apply(pd.to_numeric, errors="coerce")
    fe_X["numeric_sum"] = numeric_frame.sum(axis=1)
    fe_X["numeric_mean"] = numeric_frame.mean(axis=1)
    fe_X = fe_X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def make_preprocessor(X_df: pd.DataFrame):
        numeric_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [c for c in X_df.columns if c not in numeric_cols]
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_cols),
                ("cat", cat_pipe, categorical_cols),
            ]
        )

    is_regression = bool(pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15)
    stratify = y if (not is_regression and y.nunique(dropna=True) > 1) else None

    Xb_train, Xb_val, y_train, y_val = train_test_split(
        base_X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    Xf_train, Xf_val, _, _ = train_test_split(
        fe_X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    pre_base = make_preprocessor(Xb_train)
    pre_fe = make_preprocessor(Xf_train)
    Xb_train_t = pre_base.fit_transform(Xb_train)
    Xb_val_t = pre_base.transform(Xb_val)
    Xf_train_t = pre_fe.fit_transform(Xf_train)
    Xf_val_t = pre_fe.transform(Xf_val)

    if is_regression:
        base_model = HistGradientBoostingRegressor(random_state=42, max_iter=80, max_depth=5)
        fe_model = HistGradientBoostingRegressor(random_state=42, max_iter=80, max_depth=5)
        base_model.fit(Xb_train_t, y_train)
        fe_model.fit(Xf_train_t, y_train)

        base_pred = base_model.predict(Xb_val_t)
        fe_pred = fe_model.predict(Xf_val_t)

        y_val_num = np.asarray(pd.to_numeric(y_val, errors="coerce").fillna(0.0), dtype=float)
        non_negative_target = bool(np.nanmin(y_val_num) >= 0.0)
        if non_negative_target:
            base_metric = float(np.sqrt(mean_squared_log_error(np.clip(y_val_num, 0.0, None), np.clip(base_pred, 0.0, None))))
            fe_metric = float(np.sqrt(mean_squared_log_error(np.clip(y_val_num, 0.0, None), np.clip(fe_pred, 0.0, None))))
            better_with_fe = fe_metric < base_metric * 0.90
            reason = (
                "A/B RMSLE check (conservative gate, require >=10% gain): "
                f"baseline={base_metric:.6f}, featured={fe_metric:.6f}"
            )
        else:
            base_metric = float(np.sqrt(mean_squared_error(y_val_num, base_pred)))
            fe_metric = float(np.sqrt(mean_squared_error(y_val_num, fe_pred)))
            better_with_fe = fe_metric < base_metric * 0.985
            reason = f"A/B RMSE check: baseline={base_metric:.6f}, featured={fe_metric:.6f}"

        return better_with_fe, reason

    base_model = HistGradientBoostingClassifier(random_state=42, max_iter=80, max_depth=5)
    fe_model = HistGradientBoostingClassifier(random_state=42, max_iter=80, max_depth=5)
    base_model.fit(Xb_train_t, y_train)
    fe_model.fit(Xf_train_t, y_train)
    base_acc = float(accuracy_score(y_val, base_model.predict(Xb_val_t)))
    fe_acc = float(accuracy_score(y_val, fe_model.predict(Xf_val_t)))
    better_with_fe = fe_acc > base_acc + 0.003
    return better_with_fe, f"A/B accuracy check: baseline={base_acc:.6f}, featured={fe_acc:.6f}"


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract JSON from text with robust error handling."""
    content = text.strip()
    
    # Remove markdown code blocks
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        pass
    
    # Find the actual JSON object boundaries
    start_idx = content.find("{")
    if start_idx == -1:
        raise ValueError(f"No opening brace found in: {content[:100]}")
    
    # Find matching closing brace - count braces
    brace_count = 0
    end_idx = -1
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(content)):
        char = content[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
    
    if end_idx == -1:
        raise ValueError(f"No matching closing brace found in: {content[:200]}")
    
    json_str = content[start_idx : end_idx + 1]
    
    # Try parsing extracted JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Last attempt: normalize whitespace + remove trailing commas before closing braces/brackets
        normalized = ' '.join(json_str.split())
        normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(normalized)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            raise ValueError(
                f"JSON parsing failed. Extracted (first 300 chars): {json_str[:300]}"
            ) from e


def llm_decision_node(state: AgentGraphState, llm: Any) -> Dict[str, Any]:
    """Ask the LLM for the next action with strict structured output."""
    dataset_summary = state.get("dataset_summary", {})
    metrics = state.get("current_metrics", {})
    competition_context = _competition_context_for_prompt(state)

    prompt = (
        "You are an ML orchestration brain. Return JSON only with keys decision and reason. "
        "Allowed decisions: train, tune, try_alternative_model, finalize. "
        "Decide only from current state context. "
        f"{competition_context}"
        f"State summary: rows={dataset_summary.get('rows')}, missing_pct={dataset_summary.get('missing_pct')}, "
        f"problem_type={state.get('problem_type')}, iteration_count={state.get('iteration_count')}, "
        f"max_iterations={state.get('max_iterations')}, tuning_performed={state.get('tuning_performed')}, "
        f"best_model={state.get('best_model_name')}, metrics={metrics}."
    )

    try:
        raw_response = llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM decision node failed: {exc}") from exc

    content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    raw_text = str(content)
    data = _extract_json_object(raw_text)
    decision_obj = DecisionOutput(**data)
    original_decision = str(decision_obj.decision).strip()
    decision = original_decision
    reason = str(decision_obj.reason).strip()

    if decision not in {"train", "tune", "try_alternative_model", "finalize"}:
        raise ValueError(f"Invalid LLM decision '{decision}'")

    decision_source = "llm"
    no_improvement_rounds = int(state.get("no_improvement_rounds", 0))
    metric_delta_pct = float(state.get("metric_delta_pct", 0.0))
    tuning_performed = bool(state.get("tuning_performed", False))
    tuning_pass_count = int(state.get("tuning_pass_count", 0))
    max_tuning_passes = int(state.get("max_tuning_passes", 3) or 3)
    tuning_budget_exhausted = tuning_pass_count >= max_tuning_passes
    features_added_this_round = bool(state.get("features_added_this_round", False))
    ensemble_no_improvement_rounds = int(state.get("ensemble_no_improvement_rounds", 0))
    ensemble_improvement_pct = float(state.get("ensemble_improvement_pct", 0.0))
    min_ensemble_improvement_pct = 0.01
    min_round_improvement_pct = 0.01

    if ensemble_no_improvement_rounds >= 3 and ensemble_improvement_pct < min_ensemble_improvement_pct:
        decision = "finalize"
        reason = (
            f"Forced finalize by ensemble stopping rule: ensemble CV improvement <{min_ensemble_improvement_pct:.2f}% for three rounds."
        )
        decision_source = "llm+ensemble_stopping_override"

    if no_improvement_rounds >= 3:
        decision = "finalize"
        reason = (
            f"Forced finalize due to stagnation control: relative improvement <{min_round_improvement_pct:.2f}% for 3 consecutive rounds. "
            f"Original LLM reason: {reason}"
        )
        decision_source = "llm+stagnation_override"

    if decision == "tune" and tuning_budget_exhausted:
        decision = "finalize"
        reason = (
            f"Tuning pass budget exhausted ({tuning_pass_count}/{max_tuning_passes}); converting repeated tune request to finalize "
            f"to avoid redundant routing. Original LLM reason: {reason}"
        )
        decision_source = "llm+tuning_guard_override"

    if metric_delta_pct < min_round_improvement_pct and tuning_budget_exhausted and (not features_added_this_round):
        decision = "finalize"
        reason = (
            f"Early stopping rule triggered: CV improvement <{min_round_improvement_pct:.2f}%, tuning budget exhausted, "
            "and no new features were added in this round."
        )
        decision_source = "llm+early_stop_override"

    # Force tuning if flag set and tuning not yet performed
    if state.get("force_tune") and not tuning_budget_exhausted and decision != "finalize":
        decision = "tune"
        reason = f"Force tuning enabled. Original LLM decision: {reason}"
        decision_source = "llm+force_tune_override"

    if state.get("force_tune") and tuning_budget_exhausted:
        iteration_count = int(state.get("iteration_count", 0))
        max_iterations = int(state.get("max_iterations", 0))
        if decision == "finalize" and iteration_count < max_iterations - 1:
            decision = "try_alternative_model"
            reason = (
                "Force tuning exploration enabled: overriding finalize to try alternative model "
                "while iterations remain."
            )
            decision_source = "llm+force_explore_override"

    audit_entry = {
        "iteration": int(state.get("iteration_count", 0)),
        "prompt_summary": {
            "rows": dataset_summary.get("rows"),
            "missing_pct": dataset_summary.get("missing_pct"),
            "problem_type": state.get("problem_type"),
            "best_model": state.get("best_model_name"),
            "tuning_performed": state.get("tuning_performed"),
        },
        "raw_llm_output": raw_text,
        "parsed_decision": original_decision,
        "final_decision": decision,
        "decision_source": decision_source,
    }

    return {
        "pending_action": decision,
        "decision_source": decision_source,
        "features_added_this_round": False,
        "llm_decision_audit": [*state.get("llm_decision_audit", []), audit_entry],
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"LLM decision: {decision} | {reason}",
        ],
    }


def llm_model_strategy_node(state: AgentGraphState, llm: Any) -> Dict[str, Any]:
    dataset_summary = state.get("dataset_summary", {})
    problem_type = state.get("problem_type")
    target_column = state.get("target_column")
    target_relations = dataset_summary.get("relationship_insights", {}).get("top_target_relations", [])
    competition_context = _competition_context_for_prompt(state)

    prompt = (
        "You are an ML model strategist. Return JSON only with keys: "
        "model_candidates, hyperparameter_suggestions, reason. "
        "Select up to 5 model class names using your own judgment from scikit-learn estimators "
        "that fit this task. Do not assume a fixed menu. "
        "Order model_candidates from best expected performance to worst for this dataset. "
        "For hyperparameter_suggestions provide lightweight GridSearch-style lists for selected models. "
        "Use parameter names that belong to each selected estimator class. "
        "Return strict JSON only (no markdown, no commentary, no trailing commas). "
        "For very large datasets (>=200k rows), prioritize scalable models and avoid heavy models "
        "such as RandomForest, ExtraTrees, KNeighbors, GaussianProcess, and SVR. "
        f"{competition_context}"
        f"Context: problem_type={problem_type}, target={target_column}, rows={dataset_summary.get('rows')}, "
        f"columns={dataset_summary.get('cols')}, missing_pct={dataset_summary.get('missing_pct')}, "
        f"used_features={dataset_summary.get('used_feature_count')}, top_target_relations={target_relations}."
    )

    try:
        raw_response = llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM strategy node failed: {exc}") from exc

    content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    raw_text = str(content)
    data = _extract_json_object(raw_text)
    if "hyperparameter_suggestions" in data:
        data["hyperparameter_suggestions"] = _normalize_hyperparameter_suggestions(
            data.get("hyperparameter_suggestions")
        )
    strategy_obj = StrategyOutput(**data)

    candidates = [str(name).strip() for name in strategy_obj.model_candidates if str(name).strip()]
    deduped_candidates: list[str] = []
    seen = set()
    for name in candidates:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(name)

    if not deduped_candidates:
        raise ValueError("LLM strategy returned empty model_candidates")

    return {
        "suggested_models": deduped_candidates,
        "suggested_hyperparams": strategy_obj.hyperparameter_suggestions,
        "strategy_reason": str(strategy_obj.reason).strip(),
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"LLM strategy selected models: {', '.join(deduped_candidates)}",
        ],
    }


def llm_preprocess_strategy_node(state: AgentGraphState, llm: Any) -> Dict[str, Any]:
    dataset_summary = state.get("dataset_summary", {})
    problem_type = state.get("problem_type")
    competition_context = _competition_context_for_prompt(state)

    prompt = (
        "You are an ML preprocessing strategist. Return JSON only with keys: "
        "numeric_imputer, categorical_imputer, categorical_fill_value, encoder, test_size, "
        "validation_size_within_train, m_estimate_smoothing, reason. "
        "Choose methods using your own judgment based on the dataset. "
        "test_size and validation_size_within_train should be practical for generalization and stable evaluation. "
        "Choose a robust strategy for tabular ML generalization. "
        f"{competition_context}"
        f"Context: problem_type={problem_type}, rows={dataset_summary.get('rows')}, "
        f"columns={dataset_summary.get('cols')}, missing_pct={dataset_summary.get('missing_pct')}, "
        f"column_types={dataset_summary.get('column_types', {})}."
    )

    try:
        raw_response = llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM preprocess strategy node failed: {exc}") from exc

    content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    raw_text = str(content)
    data = _extract_json_object(raw_text)
    strategy_obj = PreprocessStrategyOutput(**data)

    strategy = {
        "numeric_imputer": str(strategy_obj.numeric_imputer).strip().lower(),
        "categorical_imputer": str(strategy_obj.categorical_imputer).strip().lower(),
        "categorical_fill_value": (
            "missing" if strategy_obj.categorical_fill_value is None else str(strategy_obj.categorical_fill_value)
        ),
        "encoder": str(strategy_obj.encoder).strip().lower(),
        "test_size": float(strategy_obj.test_size),
        "validation_size_within_train": float(strategy_obj.validation_size_within_train),
        "m_estimate_smoothing": float(strategy_obj.m_estimate_smoothing or 10.0),
    }

    return {
        "preprocess_strategy": strategy,
        "preprocess_strategy_reason": str(strategy_obj.reason).strip(),
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            (
                "LLM preprocessing strategy: "
                f"num_imputer={strategy['numeric_imputer']}, "
                f"cat_imputer={strategy['categorical_imputer']}, "
                f"encoder={strategy['encoder']}, "
                f"test_size={strategy['test_size']:.2f}, "
                f"val_within_train={strategy['validation_size_within_train']:.2f}, "
                f"m_estimate_smoothing={strategy['m_estimate_smoothing']:.2f}"
            ),
        ],
    }


def llm_feature_engineering_decision_node(state: AgentGraphState, llm: Any) -> Dict[str, Any]:
    """Ask the LLM whether to apply feature engineering based on dataset characteristics."""
    dataset_summary = state.get("dataset_summary", {})
    problem_type = state.get("problem_type")
    rows = dataset_summary.get('rows', 0)
    competition_context = _competition_context_for_prompt(state)

    prompt = (
        "You are a feature engineering strategist. Return JSON only with keys: "
        "apply_feature_engineering (boolean) and reason (string). "
        "Decide whether creating synthetic features (squared, interactions, ratios, sums) "
        "will likely improve model performance for this task. "
        "Be conservative: return YES only if you expect a meaningful gain (not a tiny change). "
        "If expected improvement is marginal or uncertain, return FALSE. "
        "Prefer YES mainly when there is strong evidence of nonlinear interactions that baseline models may miss. "
        f"{competition_context}"
        f"Context: problem_type={problem_type}, rows={rows}, "
        f"features={dataset_summary.get('cols')}, missing_pct={dataset_summary.get('missing_pct'):.1f}%, "
        f"target_relations={dataset_summary.get('relationship_insights', {}).get('top_target_relations', [])}. "
        "Do not default to YES based on dataset size alone. "
        'Return ONLY valid JSON like: {"apply_feature_engineering": true, "reason": "..."}'
    )

    try:
        raw_response = llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM feature engineering decision node failed: {exc}") from exc

    content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    raw_text = str(content)
    
    try:
        data = _extract_json_object(raw_text)
    except Exception as exc:
        # Fallback: if LLM response is malformed, default to YES for large datasets
        print(f"Warning: FE decision parsing failed, defaulting. Response: {raw_text[:200]}")
        data = {
            "apply_feature_engineering": rows > 1000,
            "reason": "Fallback decision due to parsing error"
        }
    
    try:
        decision_obj = FeatureEngineeringDecisionOutput(**data)
    except Exception as exc:
        print(f"Warning: FE decision validation failed: {exc}")
        decision_obj = FeatureEngineeringDecisionOutput(
            apply_feature_engineering=rows > 1000,
            reason="Fallback decision due to validation error"
        )

    apply_fe = bool(decision_obj.apply_feature_engineering)
    reason = str(decision_obj.reason).strip()

    reason = f"LLM confirmed feature engineering decision: {reason}"

    print(f"[LLM FE Decision] Decision: {apply_fe}, Reason: {reason}")
    
    return {
        "apply_feature_engineering": apply_fe,
        "feature_engineering_reason": reason,
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"LLM feature engineering confirmation: {'YES' if apply_fe else 'NO'} | {reason}",
        ],
    }
