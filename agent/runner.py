"""Agent runner and final report rendering."""
from __future__ import annotations

import logging
import os
import json
import time
from itertools import combinations
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_squared_log_error
import warnings
from sklearn.base import clone

from agent.graph import build_graph, get_graph_mermaid, get_graph_spec
from agent.state import AgentGraphState, initialize_state

logger = logging.getLogger(__name__)


def _load_existing_report(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _extract_primary_score_from_payload(payload: Dict[str, Any]) -> tuple[str, float, bool] | None:
    if not isinstance(payload, dict):
        return None

    problem_type = str(payload.get("problem_type") or "").lower().strip()

    holdout_metrics = payload.get("final_holdout_metrics") or {}
    if isinstance(holdout_metrics, dict):
        if problem_type == "classification" and "Accuracy" in holdout_metrics:
            try:
                return "Accuracy", float(holdout_metrics["Accuracy"]), True
            except Exception:
                pass
        for reg_metric in (() if problem_type == "classification" else ("RMSLE", "MAE", "MSE")):
            if reg_metric in holdout_metrics:
                try:
                    return reg_metric, float(holdout_metrics[reg_metric]), False
                except Exception:
                    pass

    best_model_name = str(payload.get("best_model_name") or "")
    current_metrics = payload.get("current_metrics") or {}
    if best_model_name and isinstance(current_metrics, dict) and best_model_name in current_metrics:
        metric_row = current_metrics.get(best_model_name) or {}
        if problem_type == "classification" and "CV_Accuracy" in metric_row:
            try:
                return "CV_Accuracy", float(metric_row["CV_Accuracy"]), True
            except Exception:
                pass
        for reg_cv_metric in (() if problem_type == "classification" else ("CV_RMSLE", "CV_MAE", "CV_MSE")):
            if reg_cv_metric in metric_row:
                try:
                    return reg_cv_metric, float(metric_row[reg_cv_metric]), False
                except Exception:
                    pass

    cv_scores = payload.get("cv_scores") or {}
    if best_model_name and isinstance(cv_scores, dict) and best_model_name in cv_scores:
        try:
            return "cv_mean", float((cv_scores[best_model_name] or {}).get("cv_mean")), True
        except Exception:
            pass

    return None


def _should_promote_current_run(current_payload: Dict[str, Any], existing_report_path: str) -> tuple[bool, str]:
    previous_payload = _load_existing_report(existing_report_path)
    if not previous_payload:
        return True, "No previous report found; promoting current run"

    current_problem_type = str(current_payload.get("problem_type") or "").strip().lower()
    previous_problem_type = str(previous_payload.get("problem_type") or "").strip().lower()
    current_target = str(current_payload.get("target_column") or "").strip().lower()
    previous_target = str(previous_payload.get("target_column") or "").strip().lower()

    if current_problem_type and not previous_problem_type:
        previous_payload = {**previous_payload, "problem_type": current_problem_type}
        previous_problem_type = current_problem_type
    if current_target and not previous_target:
        previous_payload = {**previous_payload, "target_column": current_target}
        previous_target = current_target

    if current_problem_type and previous_problem_type and current_problem_type != previous_problem_type:
        return True, "Problem type changed; promoting current run"
    if current_target and previous_target and current_target != previous_target:
        return True, "Target column changed; promoting current run"

    def _composite_selection_score(payload: Dict[str, Any]) -> tuple[float, str] | None:
        try:
            problem = str(payload.get("problem_type") or "").strip().lower()
            if problem != "classification":
                return None
            best_name = str(payload.get("best_model_name") or "")
            holdout = payload.get("final_holdout_metrics") or {}
            metrics = payload.get("current_metrics") or {}
            holdout_acc = float(holdout.get("Accuracy", float("nan")))
            if not np.isfinite(holdout_acc):
                return None
            metric_blob = (metrics.get(best_name) or {}) if isinstance(metrics, dict) and best_name else {}
            cv_acc = float(metric_blob.get("CV_Accuracy", holdout_acc))
            cv_std = float(metric_blob.get("CV_Accuracy_STD", 0.0))
            drift_gap = abs(float(holdout_acc - cv_acc)) if np.isfinite(cv_acc) else 0.0
            score = float(holdout_acc - (0.40 * cv_std) - (0.30 * drift_gap))
            return score, (
                f"composite={score:.6f} (holdout={holdout_acc:.6f}, cv_std={cv_std:.6f}, gap={drift_gap:.6f})"
            )
        except Exception:
            return None

    prev_comp = _composite_selection_score(previous_payload)
    curr_comp = _composite_selection_score(current_payload)
    if prev_comp is not None and curr_comp is not None:
        curr_score, curr_detail = curr_comp
        prev_score, prev_detail = prev_comp
        if curr_score > prev_score + 1e-6:
            return True, f"Improved stability-aware selection score: {curr_detail} > previous {prev_detail}"
        return False, f"Kept previous by stability-aware selection score: {curr_detail} <= previous {prev_detail}"

    previous_score = _extract_primary_score_from_payload(previous_payload)
    current_score = _extract_primary_score_from_payload(current_payload)
    if current_score is None:
        return True, "Current run score unavailable; promoting current run"
    if previous_score is None:
        return True, "Previous run score unavailable; promoting current run"

    current_metric, current_value, current_higher_is_better = current_score
    previous_metric, previous_value, previous_higher_is_better = previous_score
    if current_higher_is_better != previous_higher_is_better:
        return True, "Metric direction mismatch; promoting current run"

    tolerance = 1e-6
    if current_higher_is_better:
        if current_value > previous_value + tolerance:
            return True, f"Improved score: {current_metric} {current_value:.6f} > previous {previous_metric} {previous_value:.6f}"
        return False, f"Kept previous best: {current_metric} {current_value:.6f} <= previous {previous_metric} {previous_value:.6f}"

    if current_value < previous_value - tolerance:
        return True, f"Improved score: {current_metric} {current_value:.6f} < previous {previous_metric} {previous_value:.6f}"
    return False, f"Kept previous best: {current_metric} {current_value:.6f} >= previous {previous_metric} {previous_value:.6f}"


def _load_competition_context(path: str | None, max_chars: int = 12000) -> tuple[str, str]:
    if not path:
        return "", ""
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        return "", ""
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
    except Exception as exc:
        logger.warning("Failed to load competition description from %s: %s", resolved, exc)
        return "", ""

    if len(text) > max_chars:
        text = f"{text[:max_chars].rstrip()}\n...[truncated]"

    return text, resolved


def _to_jsonable(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return bool(value)
    except Exception:
        return default


def _score_from_objective(objective: str, y_true: np.ndarray, y_pred: np.ndarray, problem_type: str) -> float:
    obj = str(objective or "").lower()
    if problem_type == "classification":
        from sklearn.metrics import accuracy_score
        y_true_series = pd.Series(y_true).astype(str)
        y_pred_series = pd.Series(y_pred).astype(str)
        return float(accuracy_score(y_true_series, y_pred_series))
    if obj == "rmsle":
        return float(
            np.sqrt(
                mean_squared_log_error(
                    np.clip(y_true, 0.0, None),
                    np.clip(y_pred, 0.0, None),
                )
            )
        )
    if obj == "mae":
        from sklearn.metrics import mean_absolute_error

        return float(mean_absolute_error(y_true, y_pred))
    from sklearn.metrics import mean_squared_error

    return float(mean_squared_error(y_true, y_pred))


def _predict_with_optional_binary_threshold(
    model: Any,
    X: Any,
    threshold: float,
    positive_class_label: Any,
):
    if not hasattr(model, "predict_proba"):
        return model.predict(X)

    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size != 2:
        return model.predict(X)

    if positive_class_label is None or positive_class_label not in classes:
        positive_class_label = classes[-1]

    negative_class_label = classes[0] if classes[0] != positive_class_label else classes[1]
    hit = np.where(classes == positive_class_label)[0]
    if len(hit) == 0:
        return model.predict(X)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
        proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2 or proba.shape[1] <= int(hit[0]):
        return model.predict(X)

    pos_proba = proba[:, int(hit[0])]
    return np.where(pos_proba >= float(threshold), positive_class_label, negative_class_label)


def _evaluate_final_holdout_once(state: AgentGraphState) -> AgentGraphState:
    if state.get("holdout_evaluated_once"):
        return state

    model = state.get("best_model_object")
    X_holdout_transformed = state.get("X_test")
    X_holdout_raw = state.get("X_holdout_raw")
    y_holdout = state.get("y_test")
    if model is None or y_holdout is None:
        return state

    is_pipeline_model = hasattr(model, "named_steps") or hasattr(model, "base_models")
    X_holdout = X_holdout_raw if is_pipeline_model else X_holdout_transformed
    if X_holdout is None:
        return state

    problem_type = str(state.get("problem_type", ""))
    if problem_type == "classification":
        threshold = float(state.get("classification_threshold", 0.5) or 0.5)
        positive_label = state.get("positive_class_label")
        preds = np.asarray(
            _predict_with_optional_binary_threshold(
                model=model,
                X=X_holdout,
                threshold=threshold,
                positive_class_label=positive_label,
            ),
            dtype=object,
        )
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
            preds = np.asarray(model.predict(X_holdout), dtype=float)
    if state.get("problem_type") == "regression" and state.get("target_transform_applied"):
        preds = np.expm1(preds)

    y_true = np.asarray(y_holdout)
    objective = str(state.get("metric_objective", "")).lower()
    holdout_metric_value = _score_from_objective(objective, y_true, preds, problem_type=problem_type)

    metric_name = "Accuracy" if problem_type == "classification" else str(objective or "mse").upper()
    state["final_holdout_metrics"] = {metric_name: float(holdout_metric_value)}
    state["holdout_evaluated_once"] = True
    state.setdefault("reasoning_trace", []).append(
        f"Final holdout evaluated once: {metric_name}={holdout_metric_value:.6f}"
    )
    return state


def _save_run_artifacts(state: AgentGraphState) -> AgentGraphState:
    output_dir = os.path.abspath("artifacts")
    os.makedirs(output_dir, exist_ok=True)

    model_bundle_path = os.path.join(output_dir, "best_model_bundle.joblib")
    report_path = os.path.join(output_dir, "evaluation_report.json")

    bundle = {
        "model": state.get("best_model_object"),
        "preprocessor": state.get("preprocessor"),
        "target_column": state.get("target_column"),
        "problem_type": state.get("problem_type"),
        "metric_objective": state.get("metric_objective"),
        "target_transform_applied": state.get("target_transform_applied"),
    }
    dump(bundle, model_bundle_path)

    report_payload = {
        "best_model_name": state.get("best_model_name"),
        "problem_type": state.get("problem_type"),
        "target_column": state.get("target_column"),
        "metric_objective": state.get("metric_objective"),
        "cv_scores": state.get("cv_scores", {}),
        "current_metrics": state.get("current_metrics", {}),
        "final_holdout_metrics": state.get("final_holdout_metrics", {}),
        "model_params": state.get("model_params", {}).get(state.get("best_model_name", ""), {}),
        "improvement_timeline": state.get("improvement_timeline", []),
        "stagnation_log": state.get("stagnation_log", []),
        "encoding_choice": state.get("encoding_choice", ""),
        "encoding_cv_scores": state.get("encoding_cv_scores", {}),
        "feature_expansion_decisions": state.get("feature_expansion_decisions", {}),
        "ensemble_cv_score": state.get("ensemble_cv_score", None),
        "ensemble_cv_std": state.get("ensemble_cv_std", None),
        "ensemble_improvement_pct": state.get("ensemble_improvement_pct", None),
        "ridge_ensemble_cv_score": state.get("ridge_ensemble_cv_score", None),
        "ridge_ensemble_cv_std": state.get("ridge_ensemble_cv_std", None),
        "ridge_ensemble_alpha": state.get("ridge_ensemble_alpha", None),
        "ridge_ensemble_top_k": state.get("ridge_ensemble_top_k", None),
        "model_prediction_correlation": state.get("model_prediction_correlation", {}),
        "dropped_correlated_models": state.get("dropped_correlated_models", []),
        "target_transform_decision": state.get("target_transform_decision"),
        "feature_pruning_decision": state.get("feature_pruning_decision"),
        "classification_threshold": state.get("classification_threshold", 0.5),
        "positive_class_label": state.get("positive_class_label"),
        "run_timing": state.get("run_timing", {}),
    }

    should_promote, promote_reason = _should_promote_current_run(report_payload, report_path)
    if not should_promote:
        state.setdefault("reasoning_trace", []).append(
            f"Artifact promotion skipped: {promote_reason}"
        )
        state["model_artifact_path"] = model_bundle_path if os.path.exists(model_bundle_path) else ""
        state["report_artifact_path"] = report_path if os.path.exists(report_path) else ""
        return state

    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(_to_jsonable(report_payload), fp, indent=2, ensure_ascii=False)

    state["model_artifact_path"] = model_bundle_path
    state["report_artifact_path"] = report_path
    state.setdefault("reasoning_trace", []).append(
        f"Saved artifacts: model_bundle={model_bundle_path}, evaluation_report={report_path} | {promote_reason}"
    )
    return state


def _format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    def as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    lines = []
    for model_name, values in metrics.items():
        metric_text = ", ".join([f"{k}: {as_float(v):.4f}" for k, v in values.items()])
        lines.append(f"  - {model_name}: {metric_text}")
    return "\n".join(lines) if lines else "  - None"


def _apply_engineered_features_to_test(
    test_df: pd.DataFrame,
    base_columns: list,
    engineered_columns: list,
    aggregate_base_columns: list,
) -> pd.DataFrame:
    """Apply the same engineered features to test data as were applied to training."""
    test_engineered = test_df.copy()

    def safe_numeric(s):
        return pd.to_numeric(s, errors="coerce")

    for engineered_name in engineered_columns:
        if engineered_name in test_engineered.columns:
            continue

        if engineered_name.endswith("_squared"):
            base = engineered_name[: -len("_squared")]
            if base in test_engineered.columns:
                test_engineered[engineered_name] = safe_numeric(test_engineered[base]).pow(2)
            continue

        if engineered_name.endswith("_abs"):
            base = engineered_name[: -len("_abs")]
            if base in test_engineered.columns:
                test_engineered[engineered_name] = safe_numeric(test_engineered[base]).abs()
            continue

        if "_x_" in engineered_name:
            left, right = engineered_name.split("_x_", 1)
            if left in test_engineered.columns and right in test_engineered.columns:
                test_engineered[engineered_name] = safe_numeric(test_engineered[left]) * safe_numeric(test_engineered[right])
            continue

        if "_div_" in engineered_name:
            left, right = engineered_name.split("_div_", 1)
            if left in test_engineered.columns and right in test_engineered.columns:
                left_num = safe_numeric(test_engineered[left])
                right_num = safe_numeric(test_engineered[right]).replace(0, np.nan)
                test_engineered[engineered_name] = left_num / right_num
            continue

    if aggregate_base_columns:
        existing_aggregate_cols = [c for c in aggregate_base_columns if c in test_engineered.columns]
        if existing_aggregate_cols:
            numeric_frame = test_engineered[existing_aggregate_cols].apply(pd.to_numeric, errors="coerce")
            if "numeric_sum" in engineered_columns:
                test_engineered["numeric_sum"] = numeric_frame.sum(axis=1)
            if "numeric_mean" in engineered_columns:
                test_engineered["numeric_mean"] = numeric_frame.mean(axis=1)
    
    return test_engineered


def _generate_submission_csv(state: AgentGraphState) -> AgentGraphState:
    test_file_path = state.get("test_file_path")
    if not test_file_path:
        return state

    model = state.get("best_model_object")
    preprocessor = state.get("preprocessor")
    target_column = state.get("target_column")
    if model is None:
        raise ValueError("Cannot generate submission: best model missing")

    def _resolve_sample_submission_path() -> str | None:
        explicit = str(state.get("sample_submission_file_path") or "").strip()
        if explicit and os.path.exists(explicit):
            return explicit
        if test_file_path:
            candidate = os.path.join(os.path.dirname(test_file_path), "sample_submission.csv")
            if os.path.exists(candidate):
                return candidate
        return None

    def _binary_label_numeric_map(labels: list[str]) -> Dict[str, int] | None:
        normalized = {str(value).strip().lower() for value in labels if str(value).strip()}
        if len(normalized) != 2:
            return None

        positive_tokens = {"presence", "present", "yes", "y", "true", "positive", "1", "disease", "abnormal"}
        negative_tokens = {"absence", "absent", "no", "n", "false", "negative", "0", "normal", "healthy"}

        inferred_positive = [label for label in normalized if label in positive_tokens]
        inferred_negative = [label for label in normalized if label in negative_tokens]
        if len(inferred_positive) == 1 and len(inferred_negative) == 1:
            return {inferred_negative[0]: 0, inferred_positive[0]: 1}

        ordered = sorted(normalized)
        return {ordered[0]: 0, ordered[1]: 1}

    def _align_predictions_to_sample_dtype(pred_values: Any) -> Any:
        sample_path = _resolve_sample_submission_path()
        if not sample_path:
            return pred_values

        try:
            sample_df = pd.read_csv(sample_path, nrows=200)
        except Exception:
            return pred_values

        if target_column not in sample_df.columns:
            return pred_values

        sample_target = sample_df[target_column]
        sample_is_numeric = pd.api.types.is_numeric_dtype(sample_target)
        pred_series = pd.Series(pred_values)

        if sample_is_numeric:
            pred_numeric = pd.to_numeric(pred_series, errors="coerce")
            if pred_numeric.isna().any():
                train_df = state.get("raw_df")
                train_labels = []
                if isinstance(train_df, pd.DataFrame) and target_column in train_df.columns:
                    train_labels = [str(v).strip() for v in train_df[target_column].dropna().unique().tolist()]

                mapping = _binary_label_numeric_map(train_labels)
                if mapping:
                    lowered = pred_series.astype(str).str.strip().str.lower()
                    mapped = lowered.map(mapping)
                    if mapped.notna().all():
                        pred_numeric = mapped
                        state.setdefault("reasoning_trace", []).append(
                            f"Submission dtype alignment: mapped string labels to numeric using train-label map {mapping}"
                        )

            if pd.api.types.is_integer_dtype(sample_target.dtype):
                pred_numeric = pred_numeric.fillna(0).round().astype(int)
            else:
                pred_numeric = pred_numeric.astype(float)

            state.setdefault("reasoning_trace", []).append(
                f"Submission dtype alignment: cast target '{target_column}' to sample dtype {sample_target.dtype}"
            )
            return pred_numeric.to_numpy()

        return pred_values

    test_df = pd.read_csv(test_file_path)
    if test_df.empty:
        raise ValueError("Cannot generate submission from empty test dataset")

    X_submission = test_df.copy()
    if target_column in X_submission.columns:
        X_submission = X_submission.drop(columns=[target_column])

    # Apply feature engineering to test data if it was applied during training
    apply_fe = state.get("apply_feature_engineering", False)
    engineered_columns = state.get("engineered_feature_columns", [])
    engineered_base_columns = state.get("engineered_base_columns", [])
    engineered_aggregate_base_columns = state.get("engineered_aggregate_base_columns", [])
    
    print(f"[Submission Debug] apply_fe={apply_fe}, engineered_cols={len(engineered_columns)}, base_cols={engineered_base_columns}")
    
    if apply_fe and engineered_columns and engineered_base_columns:
        print(f"[Submission Debug] Applying {len(engineered_columns)} engineered features to test data")
        X_submission = _apply_engineered_features_to_test(
            X_submission,
            engineered_base_columns,
            engineered_columns,
            engineered_aggregate_base_columns,
        )
        print(f"[Submission Debug] After feature engineering: {X_submission.shape}")
    elif apply_fe:
        print(f"[Submission Debug] Feature eng enabled but columns missing! eng_cols:{engineered_columns}, base_cols:{engineered_base_columns}")

    categorical_columns = state.get("categorical_columns", [])
    if categorical_columns:
        existing_cat_cols = [col for col in categorical_columns if col in X_submission.columns]
        if existing_cat_cols:
            X_submission[existing_cat_cols] = X_submission[existing_cat_cols].astype("object")

    X_submission = X_submission.replace({pd.NA: np.nan})

    is_pipeline_model = hasattr(model, "named_steps") or hasattr(model, "base_models")
    threshold = float(state.get("classification_threshold", 0.5) or 0.5)
    positive_label = state.get("positive_class_label")
    if is_pipeline_model:
        expected_raw_cols = list(getattr(state.get("X_train_raw"), "columns", []))
        if expected_raw_cols:
            for col in expected_raw_cols:
                if col not in X_submission.columns:
                    X_submission[col] = np.nan
            X_submission = X_submission[expected_raw_cols]
            print(f"[Submission Debug] Pipeline expects {len(expected_raw_cols)} raw columns")
        if str(state.get("problem_type", "")) == "classification":
            preds = _predict_with_optional_binary_threshold(
                model=model,
                X=X_submission,
                threshold=threshold,
                positive_class_label=positive_label,
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
                preds = model.predict(X_submission)
    else:
        if preprocessor is None:
            raise ValueError("Cannot generate submission: preprocessor missing for non-pipeline model")
        expected_columns = getattr(preprocessor, "feature_names_in_", None)
        if expected_columns is not None:
            print(f"[Submission Debug] Preprocessor expects {len(expected_columns)} columns: {list(expected_columns)}")
            for col in expected_columns:
                if col not in X_submission.columns:
                    X_submission[col] = np.nan
            X_submission = X_submission[list(expected_columns)]
            print(f"[Submission Debug] After column alignment: {X_submission.shape}")

        X_transformed = preprocessor.transform(X_submission)
        print(f"[Submission Debug] After preprocessing: {X_transformed.shape}")
        if str(state.get("problem_type", "")) == "classification":
            preds = _predict_with_optional_binary_threshold(
                model=model,
                X=X_transformed,
                threshold=threshold,
                positive_class_label=positive_label,
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
                preds = model.predict(X_transformed)
    if str(state.get("problem_type", "")) == "regression" and _safe_bool(state.get("target_transform_applied", False)):
        preds = np.expm1(np.asarray(preds, dtype=float))

    preds = _align_predictions_to_sample_dtype(preds)

    if (
        str(state.get("problem_type", "")) == "classification"
        and bool(state.get("enable_pseudo_labeling", True))
        and hasattr(model, "predict_proba")
        and hasattr(model, "fit")
        and not hasattr(model, "base_models")
    ):
        try:
            pseudo_min_conf = float(state.get("pseudo_label_min_confidence", 0.995) or 0.995)
            pseudo_min_count = int(state.get("pseudo_label_min_count", 2000) or 2000)
            proba = np.asarray(model.predict_proba(X_submission if is_pipeline_model else X_transformed), dtype=float)
            if proba.ndim == 2 and proba.shape[1] == 2:
                pos_label = state.get("positive_class_label")
                classes = np.asarray(getattr(model, "classes_", []))
                if classes.size == 2:
                    if pos_label is None or pos_label not in classes:
                        pos_label = classes[-1]
                    pos_idx = int(np.where(classes == pos_label)[0][0])
                    pos_proba = proba[:, pos_idx]
                    confident_mask = (pos_proba >= pseudo_min_conf) | (pos_proba <= (1.0 - pseudo_min_conf))
                    confident_count = int(np.sum(confident_mask))
                    if confident_count >= pseudo_min_count:
                        pseudo_labels = np.where(pos_proba[confident_mask] >= 0.5, pos_label, classes[0] if classes[0] != pos_label else classes[1])
                        X_train_raw = state.get("X_train_raw")
                        y_train_raw = state.get("y_train_raw")
                        if is_pipeline_model and X_train_raw is not None and y_train_raw is not None:
                            X_aug = pd.concat([
                                X_train_raw,
                                X_submission.loc[confident_mask].copy(),
                            ], axis=0, ignore_index=True)
                            y_aug = pd.concat([
                                pd.Series(y_train_raw).reset_index(drop=True),
                                pd.Series(pseudo_labels),
                            ], axis=0, ignore_index=True)
                            pseudo_model = clone(model)
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
                                pseudo_model.fit(X_aug, y_aug)
                            preds = _predict_with_optional_binary_threshold(
                                model=pseudo_model,
                                X=X_submission,
                                threshold=threshold,
                                positive_class_label=pos_label,
                            )
                            preds = _align_predictions_to_sample_dtype(preds)
                            state.setdefault("reasoning_trace", []).append(
                                f"Pseudo-label self-training applied: augmented with {confident_count} high-confidence test rows"
                            )
        except Exception as exc:
            state.setdefault("reasoning_trace", []).append(
                f"Pseudo-label self-training skipped: {exc}"
            )

    problem_type = str(state.get("problem_type", ""))
    trained_models = state.get("trained_models") or {}
    X_val = state.get("X_val")
    y_val = state.get("y_val")
    can_try_ensemble = (
        problem_type == "regression"
        and len(trained_models) >= 2
        and X_val is not None
        and y_val is not None
    )

    if can_try_ensemble:
        try:
            y_val_arr = np.asarray(y_val, dtype=float)
            if np.nanmin(y_val_arr) >= 0.0:
                val_scores = []
                val_preds_by_name: Dict[str, np.ndarray] = {}
                for name, candidate_model in trained_models.items():
                    try:
                        val_preds = np.asarray(candidate_model.predict(X_val), dtype=float)
                        val_preds = np.clip(val_preds, 0.0, None)
                        rmsle = float(np.sqrt(mean_squared_log_error(np.clip(y_val_arr, 0.0, None), val_preds)))
                        val_scores.append((name, rmsle, candidate_model))
                        val_preds_by_name[name] = val_preds
                    except Exception:
                        continue

                if len(val_scores) >= 2:
                    val_scores.sort(key=lambda row: row[1])
                    top = val_scores[:3]
                    best_single_name, best_single_rmsle, _ = top[0]

                    inv = np.array([1.0 / max(row[1], 1e-9) for row in top], dtype=float)
                    weights = inv / inv.sum()
                    blend_val_preds = np.zeros_like(y_val_arr, dtype=float)
                    for idx, (name, _, _) in enumerate(top):
                        blend_val_preds += weights[idx] * val_preds_by_name[name]
                    blend_val_preds = np.clip(blend_val_preds, 0.0, None)
                    blend_rmsle = float(np.sqrt(mean_squared_log_error(np.clip(y_val_arr, 0.0, None), blend_val_preds)))

                    if blend_rmsle < best_single_rmsle - 0.0002:
                        blend_test_preds = np.zeros(X_transformed.shape[0], dtype=float)
                        for idx, (name, _, candidate_model) in enumerate(top):
                            test_preds = np.asarray(candidate_model.predict(X_transformed), dtype=float)
                            blend_test_preds += weights[idx] * np.clip(test_preds, 0.0, None)
                        preds = blend_test_preds
                        state.setdefault("reasoning_trace", []).append(
                            "Using validation-gated ensemble for submission: "
                            f"models={[row[0] for row in top]}, "
                            f"single_rmsle={best_single_rmsle:.6f}, ensemble_rmsle={blend_rmsle:.6f}"
                        )
                        print(
                            "[Submission Debug] Using ensemble predictions: "
                            f"single_rmsle={best_single_rmsle:.6f}, ensemble_rmsle={blend_rmsle:.6f}"
                        )
                    else:
                        print(
                            "[Submission Debug] Ensemble skipped (no val gain): "
                            f"single_rmsle={best_single_rmsle:.6f}, ensemble_rmsle={blend_rmsle:.6f}"
                        )
        except Exception as exc:
            print(f"[Submission Debug] Ensemble strategy skipped due to error: {exc}")

    summary = state.get("dataset_summary", {})
    target_is_non_negative = _safe_bool(summary.get("target_is_non_negative", False))
    if target_is_non_negative:
        neg_count = int((preds < 0).sum())
        if neg_count > 0:
            print(f"[Submission Debug] Clipping {neg_count} negative predictions to 0.0 for non-negative target")
            preds = np.clip(preds, 0.0, None)

    try:
        preds_num = np.asarray(preds, dtype=float)
        print(
            "[Submission Debug] Predictions "
            f"min={preds_num.min():.4f}, max={preds_num.max():.4f}, mean={preds_num.mean():.4f}"
        )
    except Exception:
        pred_counts = pd.Series(preds).value_counts(dropna=False).head(5).to_dict()
        print(f"[Submission Debug] Predictions label distribution (top 5): {pred_counts}")

    identifier_columns = summary.get("identifier_columns", [])
    submission = pd.DataFrame({target_column: preds})

    if "id" in test_df.columns:
        submission.insert(0, "id", test_df["id"].values)
    else:
        id_col = None
        for candidate in identifier_columns:
            if candidate in test_df.columns:
                id_col = candidate
                break

        if id_col is not None:
            submission.insert(0, "id", test_df[id_col].values)
        else:
            submission.insert(0, "id", np.arange(1, len(test_df) + 1))

    submission_file_name = state.get("submission_file_name") or "submission.csv"
    output_dir = os.path.dirname(test_file_path) or os.getcwd()
    output_path = os.path.abspath(os.path.join(output_dir, submission_file_name))
    current_payload = {
        "best_model_name": state.get("best_model_name"),
        "problem_type": state.get("problem_type"),
        "target_column": state.get("target_column"),
        "final_holdout_metrics": state.get("final_holdout_metrics", {}),
        "current_metrics": state.get("current_metrics", {}),
        "cv_scores": state.get("cv_scores", {}),
    }
    report_path = os.path.abspath(os.path.join(os.getcwd(), "artifacts", "evaluation_report.json"))
    should_promote, promote_reason = _should_promote_current_run(current_payload, report_path)

    if should_promote:
        submission.to_csv(output_path, index=False)
        state.setdefault("reasoning_trace", []).append(
            f"Generated submission CSV: {output_path} | {promote_reason}"
        )
    else:
        state.setdefault("reasoning_trace", []).append(
            f"Submission overwrite skipped: {promote_reason}"
        )
        if os.path.exists(output_path):
            state["submission_path"] = output_path
            return state

        candidate_name = os.path.splitext(submission_file_name)[0] + "_candidate.csv"
        candidate_path = os.path.abspath(os.path.join(output_dir, candidate_name))
        submission.to_csv(candidate_path, index=False)
        state["submission_path"] = candidate_path
        state.setdefault("reasoning_trace", []).append(
            f"Primary submission missing; wrote non-promoted candidate to {candidate_path}"
        )
        return state

    state["submission_path"] = output_path
    return state


def print_final_report(state: AgentGraphState) -> None:
    def as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    print("\n=== LLM Agentic ML Report ===")
    print("Dataset Summary:")
    summary = state.get("dataset_summary", {})
    print(f"  Rows: {summary.get('rows', 'N/A')}")
    print(f"  Columns: {summary.get('cols', 'N/A')}")
    print(f"  Missing %: {as_float(summary.get('missing_pct', 0.0)):.2f}")
    print(f"  Used Features: {summary.get('used_feature_count', 'N/A')}")
    
    print("\nFeature Engineering:")
    apply_fe = state.get("apply_feature_engineering", False)
    fe_reason = state.get("feature_engineering_reason", "")
    print(f"  Applied: {apply_fe}")
    if fe_reason:
        print(f"  Reason: {fe_reason}")
    engineered_count = summary.get("engineered_feature_count", 0)
    if apply_fe and engineered_count > 0:
        print(f"  Engineered Features: {engineered_count}")
        engineered_columns = summary.get("engineered_feature_columns", [])
        if engineered_columns:
            preview = ", ".join(engineered_columns[:8])
            if len(engineered_columns) > 8:
                preview += ", ..."
            print(f"  Engineered Columns: {preview}")
    
    split_sizes = summary.get("split_sizes", {})
    if split_sizes:
        print(
            "  Split Sizes: "
            f"train={split_sizes.get('train', 'N/A')}, "
            f"validation={split_sizes.get('validation', 'N/A')}, "
            f"test={split_sizes.get('test', 'N/A')}"
        )

    dropped_ids = summary.get("dropped_identifier_columns", [])
    if dropped_ids:
        print(f"  Dropped ID-like Columns: {', '.join(dropped_ids)}")

    preprocess_applied = summary.get("preprocess_applied", {})
    if preprocess_applied:
        print(
            "  Preprocess Applied: "
            f"num_imputer={preprocess_applied.get('numeric_imputer')}, "
            f"cat_imputer={preprocess_applied.get('categorical_imputer')}, "
            f"encoder={preprocess_applied.get('encoder')}, "
            f"test_size={preprocess_applied.get('test_size')}, "
            f"val_within_train={preprocess_applied.get('validation_size_within_train')}"
        )

    print("\nProblem Type:")
    print(f"  {state.get('problem_type', 'N/A')}")

    print("\nModels Tried:")
    models_tried = state.get("models_tried", [])
    estimates = state.get("training_time_estimates", {})
    params_map = state.get("model_params", {})
    if models_tried:
        for model_name in models_tried:
            estimated = estimates.get(model_name)
            actual = params_map.get(model_name, {}).get("train_time_seconds")
            if estimated is not None and actual is not None:
                print(f"  - {model_name} (est: {as_float(estimated):.2f}s, actual: {as_float(actual):.2f}s)")
            elif actual is not None:
                print(f"  - {model_name} (actual: {as_float(actual):.2f}s)")
            else:
                print(f"  - {model_name}")
    else:
        print("  - None")

    print("\nLLM Model Strategy:")
    suggested_models = state.get("suggested_models", [])
    if suggested_models:
        print(f"  Suggested Models: {', '.join(suggested_models)}")
    strategy_reason = state.get("strategy_reason")
    if strategy_reason:
        print(f"  Reason: {strategy_reason}")

    print("\nLLM Preprocessing Strategy:")
    preprocess_strategy = state.get("preprocess_strategy", {})
    if preprocess_strategy:
        print(
            "  Suggested: "
            f"num_imputer={preprocess_strategy.get('numeric_imputer')}, "
            f"cat_imputer={preprocess_strategy.get('categorical_imputer')}, "
            f"encoder={preprocess_strategy.get('encoder')}, "
            f"test_size={preprocess_strategy.get('test_size')}, "
            f"val_within_train={preprocess_strategy.get('validation_size_within_train')}"
        )
    preprocess_reason = state.get("preprocess_strategy_reason")
    if preprocess_reason:
        print(f"  Reason: {preprocess_reason}")

    print("\nEncoding Choice:")
    print(f"  Selected: {state.get('encoding_choice', 'N/A')}")
    encoding_scores = state.get("encoding_cv_scores", {})
    if encoding_scores:
        for key, value in encoding_scores.items():
            print(f"  - {key}: {as_float(value):.6f}")

    print("\nBest Model:")
    print(f"  {state.get('best_model_name', 'N/A')}")

    cv_scores = state.get("cv_scores", {})
    best_model_name = state.get("best_model_name", "")
    if best_model_name and best_model_name in cv_scores:
        cv_mean = as_float(cv_scores[best_model_name].get("cv_mean", 0.0))
        cv_std = as_float(cv_scores[best_model_name].get("cv_std", 0.0))
        objective = str(state.get("metric_objective", "mse")).upper()
        print("\nTrain CV Score:")
        print(f"  {objective}: {cv_mean:.6f} ± {cv_std:.6f}")

    holdout_metrics = state.get("final_holdout_metrics", {})
    if holdout_metrics:
        print("\nHoldout Test Score:")
        for k, v in holdout_metrics.items():
            print(f"  {k}: {as_float(v):.6f}")

    ensemble_cv_score = state.get("ensemble_cv_score")
    if ensemble_cv_score is not None and np.isfinite(as_float(ensemble_cv_score)):
        print("\nEnsemble CV:")
        objective = str(state.get("metric_objective", "mse")).upper()
        print(
            f"  {objective}: {as_float(ensemble_cv_score):.6f} ± {as_float(state.get('ensemble_cv_std', 0.0)):.6f} "
            f"(improvement={as_float(state.get('ensemble_improvement_pct', 0.0)):.4f}%)"
        )

    ridge_cv_score = state.get("ridge_ensemble_cv_score")
    if ridge_cv_score is not None and np.isfinite(as_float(ridge_cv_score)):
        print("\nRank-Ridge Ensemble CV:")
        objective = str(state.get("metric_objective", "mse")).upper()
        print(
            f"  {objective}: {as_float(ridge_cv_score):.6f} ± {as_float(state.get('ridge_ensemble_cv_std', 0.0)):.6f} "
            f"(alpha={as_float(state.get('ridge_ensemble_alpha', 0.0)):.2f}, top_k={int(as_float(state.get('ridge_ensemble_top_k', 0.0))):d})"
        )

    print("\nFinal Metrics:")
    eval_split = state.get("metric_evaluation_split")
    if eval_split:
        print(f"  Evaluated On: {eval_split}")
    print(_format_metrics(state.get("current_metrics", {})))

    print("\nOverfitting Check:")
    overfitting_report = state.get("overfitting_report", {})
    if overfitting_report:
        for model_name, details in overfitting_report.items():
            gap = as_float(details.get("gap", 0.0))
            overfit = _safe_bool(details.get("overfitting", False))
            print(
                f"  - {model_name}: gap={gap:.4f}, overfitting={overfit}"
            )
    else:
        print("  - Not available")

    print("\nDecision Engine:")
    print("  Orchestrator: LangGraph StateGraph")
    decision_source = state.get("decision_source")
    if decision_source:
        print(f"  Last Decision Source: {decision_source}")

    print("\nHyperparameters:")
    params = state.get("model_params", {}).get(state.get("best_model_name", ""), {})
    if params:
        for key, value in params.items():
            print(f"  - {key}: {value}")
    else:
        print("  - Not available")

    print("\nFeature Importance (Top 5):")
    feature_importance = state.get("final_feature_importance", [])
    if feature_importance:
        for row in feature_importance:
            print(f"  - {row['feature']}: {as_float(row.get('importance', 0.0)):.4f}")
    else:
        print("  - Not available for this model")

    print("\nFeature Expansion Decisions:")
    expansion_decisions = state.get("feature_expansion_decisions", {})
    if expansion_decisions:
        for model_name, info in expansion_decisions.items():
            print(
                f"  - {model_name}: family={info.get('model_family')}, "
                f"inverse={info.get('inverse_features')}, rf_embedding={info.get('random_trees_embedding')}"
            )
    else:
        print("  - Not available")

    print("\nModel Diversity Correlation Matrix:")
    corr_matrix = state.get("model_prediction_correlation", {})
    if corr_matrix:
        for left, row in corr_matrix.items():
            pretty = ", ".join([f"{right}:{as_float(val):.3f}" for right, val in row.items()])
            print(f"  - {left}: {pretty}")
    else:
        print("  - Not available")
    dropped_corr = state.get("dropped_correlated_models", [])
    if dropped_corr:
        print(f"  Dropped (corr>0.95): {', '.join(dropped_corr)}")

    submission_path = state.get("submission_path")
    if submission_path:
        print("\nSubmission File:")
        print(f"  {submission_path}")

    print("\nColumn Relationship Insights:")
    relationships = summary.get("relationship_insights", {})
    target_relations = relationships.get("top_target_relations", [])
    numeric_corrs = relationships.get("top_numeric_correlations", [])

    print("  Target Relations (Top 5):")
    if target_relations:
        for row in target_relations:
            feature = row.get("feature", row.get("pair", "unknown"))
            score = float(row.get("score", 0.0))
            print(f"    - {feature}: {score:.4f}")
    else:
        print("    - Not available")

    print("  Numeric Correlations (Top 5):")
    if numeric_corrs:
        for row in numeric_corrs:
            print(f"    - {row.get('pair', 'unknown')}: {float(row.get('score', 0.0)):.4f}")
    else:
        print("    - Not available")

    print("\nFull Reasoning Trace:")
    trace = state.get("reasoning_trace", [])
    if trace:
        for idx, message in enumerate(trace, start=1):
            print(f"  {idx}. {message}")
    else:
        print("  - No reasoning trace captured")

    print("\nWorkflow Trace:")
    workflow_trace = state.get("workflow_trace", [])
    if workflow_trace:
        print(f"  {' -> '.join(workflow_trace)}")
    else:
        print("  - Not available")

    print("\nLLM Decision Audit:")
    decision_audit = state.get("llm_decision_audit", [])
    if decision_audit:
        for idx, row in enumerate(decision_audit, start=1):
            parsed_decision = row.get("parsed_decision", "N/A")
            final_decision = row.get("final_decision", "N/A")
            source = row.get("decision_source", "N/A")
            print(
                f"  {idx}. parsed={parsed_decision}, final={final_decision}, source={source}"
            )
    else:
        print("  - No LLM decisions recorded")

    if state.get("show_langgraph"):
        graph_spec = get_graph_spec()
        print("\nLangGraph Full Structure:")
        print(f"  Nodes: {', '.join(graph_spec.get('nodes', []))}")
        print("  Edges:")
        for src, dst in graph_spec.get("edges", []):
            print(f"    - {src} -> {dst}")

        conditional = graph_spec.get("conditional_edges", {})
        if conditional:
            print("  Conditional Edges:")
            for source_node, mapping in conditional.items():
                for route_key, target_node in mapping.items():
                    print(f"    - {source_node} [{route_key}] -> {target_node}")

        safeguards = graph_spec.get("safeguards", [])
        if safeguards:
            print("  Safeguards:")
            for rule in safeguards:
                print(f"    - {rule}")

        print("\nLangGraph Mermaid:")
        print(get_graph_mermaid())

    print("\nImprovement Timeline:")
    timeline = state.get("improvement_timeline", [])
    if timeline:
        for row in timeline:
            print(
                f"  - iter={row.get('iteration')} model={row.get('best_model')} "
                f"score={float(row.get('best_score', 0.0)):.6f} delta_pct={float(row.get('delta_pct', 0.0)):.4f}%"
            )
    else:
        print("  - Not available")

    print("\nStagnation Detection Log:")
    stagnation_log = state.get("stagnation_log", [])
    if stagnation_log:
        for row in stagnation_log:
            print(f"  - {row}")
    else:
        print("  - Not available")

    print("\nTarget Transformation Decision:")
    print(f"  {state.get('target_transform_decision', 'none')}")

    print("\nFeature Pruning Decision:")
    print(f"  {state.get('feature_pruning_decision', 'none')}")

    model_artifact_path = state.get("model_artifact_path")
    report_artifact_path = state.get("report_artifact_path")
    if model_artifact_path or report_artifact_path:
        print("\nArtifacts:")
        if model_artifact_path:
            print(f"  Model bundle: {model_artifact_path}")
        if report_artifact_path:
            print(f"  Evaluation report: {report_artifact_path}")

    run_timing = state.get("run_timing", {})
    if isinstance(run_timing, dict) and run_timing:
        print("\nRun Timing:")
        phase_order = [
            ("workflow_seconds", "workflow"),
            ("holdout_eval_seconds", "holdout_eval"),
            ("submission_seconds", "submission"),
            ("artifact_save_seconds", "artifact_save"),
            ("total_seconds", "total"),
        ]
        for key, label in phase_order:
            if key in run_timing:
                try:
                    value = float(run_timing.get(key, 0.0))
                    print(f"  {label}: {value:.2f}s")
                except Exception:
                    pass


def run_agent(
    file_path: str,
    target_column: str,
    llm: Any,
    max_iterations: int = 5,
    force_tune: bool = False,
    metric_objective: str = "auto",
    test_file_path: str | None = None,
    sample_submission_file_path: str | None = None,
    competition_description_path: str | None = "data/data_description.txt",
    submission_file_name: str = "submission.csv",
    show_langgraph: bool = False,
) -> AgentGraphState:
    run_start = time.perf_counter()
    logger.info("Initializing agent state")
    state = initialize_state(
        file_path=file_path,
        target_column=target_column,
        max_iterations=max_iterations,
    )
    if test_file_path:
        state["test_file_path"] = test_file_path
    if sample_submission_file_path:
        state["sample_submission_file_path"] = sample_submission_file_path
    competition_context, competition_context_source = _load_competition_context(competition_description_path)
    if competition_context:
        state["competition_context"] = competition_context
        state["competition_context_source"] = competition_context_source
        state.setdefault("reasoning_trace", []).append(
            f"Loaded temporary competition context from: {competition_context_source}"
        )
    state["submission_file_name"] = submission_file_name
    state["show_langgraph"] = show_langgraph
    state["metric_objective"] = "" if str(metric_objective).lower() == "auto" else str(metric_objective).lower()
    if force_tune:
        state["force_tune"] = True
        logger.info("Force tuning enabled: agent will prioritize tuning")
    
    graph = build_graph(llm)

    logger.info("Executing LangGraph workflow")
    phase_start = time.perf_counter()
    final_state = graph.invoke(state)
    workflow_seconds = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    final_state = _evaluate_final_holdout_once(final_state)
    holdout_eval_seconds = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    final_state = _generate_submission_csv(final_state)
    submission_seconds = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    final_state = _save_run_artifacts(final_state)
    artifact_save_seconds = time.perf_counter() - phase_start
    total_seconds = time.perf_counter() - run_start

    final_state["run_timing"] = {
        "workflow_seconds": round(float(workflow_seconds), 3),
        "holdout_eval_seconds": round(float(holdout_eval_seconds), 3),
        "submission_seconds": round(float(submission_seconds), 3),
        "artifact_save_seconds": round(float(artifact_save_seconds), 3),
        "total_seconds": round(float(total_seconds), 3),
    }

    if int(final_state.get("iteration_count", 0)) >= max_iterations:
        final_state.setdefault("reasoning_trace", []).append("Stopped due to max iteration limit")

    return final_state
