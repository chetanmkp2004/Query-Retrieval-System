"""Preprocessing tool."""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Type

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class StateInput(BaseModel):
    state: Dict[str, Any] = Field(..., description="Current mutable agent state")


class PreprocessTool(BaseTool):
    name: str = "preprocess_data"
    description: str = "Apply preprocessing and create train-test split without leakage"
    args_schema: ClassVar[Type[BaseModel]] = StateInput
    MAX_ROWS_TARGET_TRANSFORM_CV: int = 120000
    MAX_ROWS_FEATURE_PRUNING: int = 80000
    CV_REPEATS_CLASSIFICATION: int = 2
    CV_REPEATS_REGRESSION: int = 2

    @staticmethod
    def _resolve_metric_objective(problem_type: str, summary: Dict[str, Any], requested: str) -> str:
        req = str(requested or "").strip().lower()
        if problem_type == "classification":
            return "accuracy"
        if req in {"rmsle", "mse", "mae"}:
            return req
        return "rmsle" if bool(summary.get("target_is_non_negative", False)) else "mse"

    @staticmethod
    def _score_from_objective(objective: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        obj = objective.lower()
        if obj == "rmsle":
            return -float(
                np.sqrt(
                    mean_squared_log_error(
                        np.clip(y_true, 0.0, None),
                        np.clip(y_pred, 0.0, None),
                    )
                )
            )
        if obj == "mae":
            return -float(mean_absolute_error(y_true, y_pred))
        return -float(mean_squared_error(y_true, y_pred))

    def _regression_cv_objective_score(self, X_df: pd.DataFrame, y_series: pd.Series, objective: str, use_log: bool) -> float:
        if len(X_df) > self.MAX_ROWS_TARGET_TRANSFORM_CV:
            sampled_idx = X_df.sample(n=self.MAX_ROWS_TARGET_TRANSFORM_CV, random_state=42).index
            X_df = X_df.loc[sampled_idx]
            y_series = y_series.loc[sampled_idx]

        model = HistGradientBoostingRegressor(random_state=42, max_depth=6, max_iter=200)
        numeric_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [col for col in X_df.columns if col not in numeric_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_cols,
                ),
            ]
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores: List[float] = []
        y_values = pd.to_numeric(y_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        for train_idx, valid_idx in cv.split(X_df):
            X_tr = X_df.iloc[train_idx]
            X_va = X_df.iloc[valid_idx]
            y_tr = y_values[train_idx]
            y_va = y_values[valid_idx]

            if use_log:
                y_tr_model = np.log1p(np.clip(y_tr, 0.0, None))
            else:
                y_tr_model = y_tr

            X_tr_t = preprocessor.fit_transform(X_tr)
            X_va_t = preprocessor.transform(X_va)
            model.fit(X_tr_t, y_tr_model)

            preds = model.predict(X_va_t)
            if use_log:
                preds = np.expm1(preds)

            fold_scores.append(self._score_from_objective(objective, y_va, preds))

        return float(np.mean(fold_scores)) if fold_scores else float("-inf")

    def _decide_target_transform(
        self,
        X_train_raw: pd.DataFrame,
        y_train_raw: pd.Series,
        problem_type: str,
        metric_objective: str,
    ) -> tuple[bool, float, Dict[str, float], str]:
        if problem_type != "regression":
            return False, 0.0, {}, "not_applicable"

        y_numeric = pd.to_numeric(y_train_raw, errors="coerce").dropna()
        if y_numeric.empty:
            return False, 0.0, {}, "not_applicable"

        skewness = float(y_numeric.skew())
        if skewness <= 1.0:
            return False, skewness, {}, "skew_below_threshold"

        baseline_score = self._regression_cv_objective_score(
            X_train_raw,
            y_train_raw,
            objective=metric_objective,
            use_log=False,
        )
        transformed_score = self._regression_cv_objective_score(
            X_train_raw,
            y_train_raw,
            objective=metric_objective,
            use_log=True,
        )

        comparison = {
            "baseline_cv_score": float(baseline_score),
            "log1p_cv_score": float(transformed_score),
        }
        improvement_pct = float(((transformed_score - baseline_score) / max(abs(baseline_score), 1e-12)) * 100.0)
        comparison["improvement_pct"] = improvement_pct
        if improvement_pct > 1.0:
            return True, skewness, comparison, "applied_log1p"
        return False, skewness, comparison, "kept_raw_target"

    def _prune_engineered_features(
        self,
        X_train_raw: pd.DataFrame,
        y_train_raw: pd.Series,
        engineered_columns: List[str],
        problem_type: str,
    ) -> tuple[pd.DataFrame, List[str], List[str], str]:
        existing_engineered = [col for col in engineered_columns if col in X_train_raw.columns]
        if len(existing_engineered) < 3:
            return X_train_raw, existing_engineered, [], "skipped_not_enough_engineered_features"

        if len(X_train_raw) > self.MAX_ROWS_FEATURE_PRUNING:
            sampled_idx = X_train_raw.sample(n=self.MAX_ROWS_FEATURE_PRUNING, random_state=42).index
            X_train_raw_prune = X_train_raw.loc[sampled_idx]
            y_train_raw_prune = y_train_raw.loc[sampled_idx]
        else:
            X_train_raw_prune = X_train_raw
            y_train_raw_prune = y_train_raw

        numeric_cols = X_train_raw.select_dtypes(include=["number"]).columns.tolist()
        usable_engineered = [col for col in existing_engineered if col in numeric_cols]
        if len(usable_engineered) < 3:
            return X_train_raw, existing_engineered, [], "skipped_non_numeric_engineered_features"

        stratify = (
            y_train_raw_prune
            if problem_type == "classification" and y_train_raw_prune.nunique(dropna=True) > 1
            else None
        )
        try:
            X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
                X_train_raw_prune,
                y_train_raw_prune,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )
        except ValueError:
            X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
                X_train_raw_prune,
                y_train_raw_prune,
                test_size=0.2,
                random_state=42,
                stratify=None,
            )

        use_cols = [col for col in numeric_cols if col in X_sub_train.columns]
        if problem_type == "classification":
            model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1, max_depth=16)
        else:
            model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1, max_depth=16)

        model.fit(X_sub_train[use_cols].fillna(0.0), y_sub_train)
        perm = permutation_importance(
            model,
            X_sub_val[use_cols].fillna(0.0),
            y_sub_val,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )

        feature_importance = {
            feature: float(score)
            for feature, score in zip(use_cols, perm.importances_mean)
        }

        ranked_engineered = sorted(
            usable_engineered,
            key=lambda col: feature_importance.get(col, 0.0),
        )
        harmful = [col for col in ranked_engineered if float(feature_importance.get(col, 0.0)) <= 0.0]
        if harmful:
            max_drop = max(1, int(np.floor(len(ranked_engineered) * 0.20)))
            dropped = harmful[:max_drop]
            pruning_reason = f"dropped_non_positive_importance={len(dropped)}"
        else:
            dropped = []
            pruning_reason = "kept_all_engineered_non_negative_importance"
        kept = [col for col in existing_engineered if col not in dropped]
        pruned = X_train_raw.drop(columns=dropped, errors="ignore")
        return pruned, kept, dropped, pruning_reason

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = state.get("raw_df")
        target_column = state.get("target_column")
        if df is None:
            raise ValueError("raw_df missing in state")
        if not target_column:
            raise ValueError("target_column missing in state")

        working_df = df.dropna(subset=[target_column]).copy()
        if working_df.empty:
            raise ValueError("No rows left after dropping missing target values")

        X_all = working_df.drop(columns=[target_column])
        y_all = working_df[target_column]

        summary = state.get("dataset_summary", {})
        identifier_columns = [col for col in summary.get("identifier_columns", []) if col in X_all.columns]
        if identifier_columns:
            X_all = X_all.drop(columns=identifier_columns)
            state.setdefault("reasoning_trace", []).append(
                f"Dropped identifier-like columns before modeling: {', '.join(identifier_columns)}"
            )

        problem_type = state.get("problem_type")
        metric_objective = self._resolve_metric_objective(
            problem_type=problem_type,
            summary=summary,
            requested=str(state.get("metric_objective", "")),
        )
        state["metric_objective"] = metric_objective

        row_indices = X_all.index.to_numpy()
        existing_train_idx = state.get("train_row_indices") or []
        existing_holdout_idx = state.get("holdout_row_indices") or []

        can_reuse_split = bool(existing_train_idx and existing_holdout_idx)
        if can_reuse_split:
            all_idx_set = set(int(v) for v in row_indices.tolist())
            train_set = set(int(v) for v in existing_train_idx)
            holdout_set = set(int(v) for v in existing_holdout_idx)
            can_reuse_split = train_set.issubset(all_idx_set) and holdout_set.issubset(all_idx_set)

        if can_reuse_split:
            train_indices = np.array(existing_train_idx, dtype=int)
            holdout_indices = np.array(existing_holdout_idx, dtype=int)
        else:
            stratify = y_all if problem_type == "classification" and y_all.nunique(dropna=True) > 1 else None
            train_indices, holdout_indices = train_test_split(
                row_indices,
                test_size=min(max(float(state.get("preprocess_strategy", {}).get("test_size", 0.2)), 0.1), 0.3),
                random_state=42,
                stratify=stratify,
            )

        X_train_raw = X_all.loc[train_indices].copy()
        y_train_raw = y_all.loc[train_indices].copy()
        X_holdout_raw = X_all.loc[holdout_indices].copy()
        y_holdout = y_all.loc[holdout_indices].copy()

        engineered_columns = [str(col) for col in state.get("engineered_feature_columns", [])]

        suspicious_dropped: List[str] = []
        if engineered_columns:
            existing_eng = [c for c in engineered_columns if c in X_train_raw.columns]
            if existing_eng:
                y_for_corr = pd.Series(y_train_raw)
                if problem_type == "classification":
                    y_codes = y_for_corr.astype("category").cat.codes
                    y_numeric = pd.Series(y_codes, index=y_for_corr.index, dtype=float)
                else:
                    y_numeric = pd.to_numeric(y_for_corr, errors="coerce")

                for col in existing_eng:
                    try:
                        x_num = pd.to_numeric(X_train_raw[col], errors="coerce")
                        valid = x_num.notna() & y_numeric.notna()
                        if int(valid.sum()) < 500:
                            continue
                        corr = float(abs(np.corrcoef(x_num[valid], y_numeric[valid])[0, 1]))
                        if np.isfinite(corr) and corr >= 0.995:
                            suspicious_dropped.append(col)
                    except Exception:
                        continue

                if suspicious_dropped:
                    X_train_raw = X_train_raw.drop(columns=suspicious_dropped, errors="ignore")
                    X_holdout_raw = X_holdout_raw.drop(columns=suspicious_dropped, errors="ignore")
                    state["raw_df"] = state["raw_df"].drop(columns=suspicious_dropped, errors="ignore")
                    state.setdefault("reasoning_trace", []).append(
                        f"Dropped leakage-suspect engineered columns by high target correlation: {', '.join(suspicious_dropped)}"
                    )

        pruned_train_raw, kept_engineered, dropped_engineered, pruning_reason = self._prune_engineered_features(
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            engineered_columns=engineered_columns,
            problem_type=problem_type,
        )
        X_train_raw = pruned_train_raw
        if dropped_engineered:
            X_holdout_raw = X_holdout_raw.drop(columns=dropped_engineered, errors="ignore")
            state["raw_df"] = state["raw_df"].drop(columns=dropped_engineered, errors="ignore")
        state["engineered_feature_columns"] = kept_engineered
        state["engineered_feature_count"] = len(kept_engineered)
        state["feature_pruning_decision"] = (
            f"kept={len(kept_engineered)}, dropped={len(dropped_engineered)}, reason={pruning_reason}"
        )

        apply_transform, skewness, transform_comparison, transform_decision = self._decide_target_transform(
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            problem_type=problem_type,
            metric_objective=metric_objective,
        )
        if apply_transform:
            y_train_model = np.log1p(np.clip(pd.to_numeric(y_train_raw, errors="coerce").fillna(0.0), 0.0, None))
            y_train_model = pd.Series(y_train_model, index=y_train_raw.index)
        else:
            y_train_model = y_train_raw

        numeric_cols: List[str] = X_train_raw.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols: List[str] = [col for col in X_train_raw.columns if col not in numeric_cols]
        if categorical_cols:
            X_train_raw[categorical_cols] = X_train_raw[categorical_cols].astype("object")
            X_holdout_raw[categorical_cols] = X_holdout_raw[categorical_cols].astype("object")

        X_train_raw = X_train_raw.replace({pd.NA: np.nan})
        X_holdout_raw = X_holdout_raw.replace({pd.NA: np.nan})

        preprocess_strategy = state.get("preprocess_strategy", {})

        numeric_imputer = str(preprocess_strategy.get("numeric_imputer", "mean")).lower()
        if numeric_imputer not in {"mean", "median"}:
            numeric_imputer = "mean"

        categorical_imputer = str(preprocess_strategy.get("categorical_imputer", "most_frequent")).lower()
        if categorical_imputer not in {"most_frequent", "constant"}:
            categorical_imputer = "most_frequent"

        categorical_fill_value = str(preprocess_strategy.get("categorical_fill_value", "missing"))

        encoder_name = str(preprocess_strategy.get("encoder", "onehot")).lower()
        if encoder_name not in {"onehot", "ordinal"}:
            encoder_name = "onehot"

        test_size = float(preprocess_strategy.get("test_size", 0.2))
        validation_size_within_train = float(preprocess_strategy.get("validation_size_within_train", 0.2))
        test_size = min(max(test_size, 0.1), 0.3)
        validation_size_within_train = min(max(validation_size_within_train, 0.1), 0.3)
        m_smoothing = float(preprocess_strategy.get("m_estimate_smoothing", state.get("mestimate_smoothing", 10.0)))
        if m_smoothing <= 0:
            m_smoothing = 10.0

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=numeric_imputer)),
                ("scaler", StandardScaler()),
            ]
        )

        if encoder_name == "ordinal":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        categorical_imputer_kwargs: Dict[str, Any] = {"strategy": categorical_imputer}
        if categorical_imputer == "constant":
            categorical_imputer_kwargs["fill_value"] = categorical_fill_value

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(**categorical_imputer_kwargs)),
                ("encoder", encoder),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
                ("cat", categorical_pipe, categorical_cols),
            ]
        )

        X_train_trans = preprocessor.fit_transform(X_train_raw)
        X_holdout_trans = preprocessor.transform(X_holdout_raw)

        feature_names = preprocessor.get_feature_names_out().tolist()

        existing_cv_splits = state.get("cv_split_indices") or []
        n_train = len(y_train_raw)
        if existing_cv_splits and all(
            isinstance(pair, (list, tuple)) and len(pair) == 2
            for pair in existing_cv_splits
        ):
            cv_split_indices = [
                (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
                for tr, va in existing_cv_splits
            ]
            can_reuse_cv = all(
                ((int(tr.max()) if tr.size else -1) < n_train and (int(va.max()) if va.size else -1) < n_train)
                for tr, va in cv_split_indices
            )
        else:
            cv_split_indices = []
            can_reuse_cv = False

        if not can_reuse_cv:
            if problem_type == "classification":
                cv_split_indices = []
                repeats = int(state.get("cv_repeats_classification", self.CV_REPEATS_CLASSIFICATION) or self.CV_REPEATS_CLASSIFICATION)
                repeats = max(1, min(repeats, 3))
                for repeat_idx in range(repeats):
                    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + repeat_idx)
                    cv_split_indices.extend(list(splitter.split(X_train_raw, y_train_raw)))
            else:
                cv_split_indices = []
                repeats = int(state.get("cv_repeats_regression", self.CV_REPEATS_REGRESSION) or self.CV_REPEATS_REGRESSION)
                repeats = max(1, min(repeats, 3))
                for repeat_idx in range(repeats):
                    splitter = KFold(n_splits=5, shuffle=True, random_state=42 + repeat_idx)
                    cv_split_indices.extend(list(splitter.split(X_train_raw, y_train_raw)))

        state["X_train"] = X_train_trans
        state["X_val"] = None
        state["X_test"] = X_holdout_trans
        state["X_train_raw"] = X_train_raw
        state["X_holdout_raw"] = X_holdout_raw
        state["y_train"] = y_train_model
        state["y_val"] = None
        state["y_test"] = y_holdout
        state["y_train_raw"] = y_train_raw
        state["y_holdout"] = y_holdout
        state["preprocessor"] = preprocessor
        state["transformed_feature_names"] = feature_names
        state["numeric_columns"] = numeric_cols
        state["categorical_columns"] = categorical_cols
        state["train_row_indices"] = [int(v) for v in train_indices.tolist()]
        state["holdout_row_indices"] = [int(v) for v in holdout_indices.tolist()]
        state["cv_split_indices"] = [
            (tr.tolist(), va.tolist())
            for tr, va in cv_split_indices
        ]
        state["preprocessing_done"] = True
        state["target_transform_applied"] = bool(apply_transform)
        state["mestimate_smoothing"] = float(m_smoothing)
        state["target_skewness"] = float(skewness)
        state["target_transform_comparison"] = transform_comparison
        state["target_transform_decision"] = transform_decision
        state["dataset_summary"]["used_feature_count"] = len(X_train_raw.columns)
        state["dataset_summary"]["dropped_identifier_columns"] = identifier_columns
        state["dataset_summary"]["split_sizes"] = {
            "train": int(len(y_train_raw)),
            "holdout": int(len(y_holdout)),
        }
        state["dataset_summary"]["preprocess_applied"] = {
            "numeric_imputer": numeric_imputer,
            "categorical_imputer": categorical_imputer,
            "encoder": encoder_name,
            "test_size": round(test_size, 3),
            "validation_size_within_train": round(validation_size_within_train, 3),
            "cv_folds": int(len(cv_split_indices)),
            "metric_objective": metric_objective,
            "m_estimate_smoothing": float(m_smoothing),
        }
        state["dataset_summary"]["engineered_feature_count"] = int(state.get("engineered_feature_count", 0))
        state["dataset_summary"]["engineered_feature_columns"] = state.get("engineered_feature_columns", [])
        state.setdefault("reasoning_trace", []).append(
            (
                "Preprocessing complete with frozen holdout + fixed 5-fold CV "
                f"[train={len(y_train_raw)}, holdout={len(y_holdout)}; "
                f"objective={metric_objective}; num_imputer={numeric_imputer}, "
                f"cat_imputer={categorical_imputer}, encoder={encoder_name}]"
            )
        )
        state.setdefault("reasoning_trace", []).append(
            f"Feature pruning decision: {state.get('feature_pruning_decision', 'none')}"
        )
        state.setdefault("reasoning_trace", []).append(
            f"Target transform decision: {transform_decision} (skewness={skewness:.4f})"
        )
        return state
