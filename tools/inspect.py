"""Dataset inspection tool."""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Type

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class StateInput(BaseModel):
    state: Dict[str, Any] = Field(..., description="Current mutable agent state")


class InspectDatasetTool(BaseTool):
    name: str = "inspect_dataset"
    description: str = "Load CSV and create dataset summary metadata"
    args_schema: ClassVar[Type[BaseModel]] = StateInput

    @staticmethod
    def _detect_identifier_columns(df: pd.DataFrame, target_column: str) -> List[str]:
        identifier_columns: List[str] = []
        row_count = len(df)
        if row_count == 0:
            return identifier_columns

        for col in df.columns:
            if col == target_column:
                continue

            col_lower = col.strip().lower()
            name_hint = (
                col_lower == "id"
                or col_lower.endswith("_id")
                or col_lower in {"index", "uuid", "record_id"}
            )

            unique_ratio = float(df[col].nunique(dropna=True) / row_count)
            unique_like = unique_ratio >= 0.98

            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            monotonic_like = False
            if is_numeric:
                series = df[col].dropna()
                if not series.empty:
                    monotonic_like = bool(series.is_monotonic_increasing or series.is_monotonic_decreasing)

            if name_hint or (unique_like and is_numeric and monotonic_like):
                identifier_columns.append(col)

        return identifier_columns

    @staticmethod
    def _compute_relationship_insights(
        df: pd.DataFrame,
        target_column: str,
        numeric_cols: List[str],
    ) -> Dict[str, Any]:
        insights: Dict[str, Any] = {
            "top_numeric_correlations": [],
            "top_target_relations": [],
        }

        numeric_no_target = [c for c in numeric_cols if c != target_column]

        if len(numeric_no_target) >= 2:
            corr_df = df[numeric_no_target].corr(numeric_only=True).abs()
            pairs = []
            for i, col_i in enumerate(corr_df.columns):
                for j in range(i + 1, len(corr_df.columns)):
                    col_j = corr_df.columns[j]
                    corr_val = corr_df.iloc[i, j]
                    if pd.notna(corr_val):
                        pairs.append({"pair": f"{col_i} ~ {col_j}", "score": float(corr_val)})
            pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)[:5]
            insights["top_numeric_correlations"] = pairs

        target = df[target_column]
        if pd.api.types.is_numeric_dtype(target):
            if target_column in numeric_cols and numeric_no_target:
                corr_with_target = (
                    df[numeric_no_target + [target_column]]
                    .corr(numeric_only=True)[target_column]
                    .drop(labels=[target_column], errors="ignore")
                    .abs()
                    .sort_values(ascending=False)
                )
                insights["top_target_relations"] = [
                    {"feature": feature, "score": float(score)}
                    for feature, score in corr_with_target.head(5).items()
                    if pd.notna(score)
                ]
        else:
            target_no_na = target.dropna()
            if not target_no_na.empty and numeric_no_target:
                class_counts = target_no_na.value_counts()
                if len(class_counts) > 1:
                    scores = []
                    for col in numeric_no_target:
                        series = df[[col, target_column]].dropna()
                        if series.empty:
                            continue
                        global_std = float(series[col].std())
                        if np.isclose(global_std, 0.0):
                            continue
                        class_means = series.groupby(target_column)[col].mean()
                        spread_score = float(class_means.std() / global_std)
                        scores.append({"feature": col, "score": spread_score})
                    insights["top_target_relations"] = sorted(
                        scores,
                        key=lambda x: x["score"],
                        reverse=True,
                    )[:5]

        return insights

    @staticmethod
    def _compute_adversarial_validation_auc(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        try:
            train_X = train_df.drop(columns=[target_column], errors="ignore").copy()
            test_X = test_df.drop(columns=[target_column], errors="ignore").copy()
            common_cols = [c for c in train_X.columns if c in test_X.columns]
            if not common_cols:
                return {"enabled": False, "reason": "no_common_columns"}

            train_X = train_X[common_cols]
            test_X = test_X[common_cols]

            max_rows_each = 50000
            if len(train_X) > max_rows_each:
                train_X = train_X.sample(n=max_rows_each, random_state=42)
            if len(test_X) > max_rows_each:
                test_X = test_X.sample(n=max_rows_each, random_state=42)

            X_adv = pd.concat([train_X, test_X], axis=0, ignore_index=True)
            y_adv = np.concatenate([
                np.zeros(len(train_X), dtype=int),
                np.ones(len(test_X), dtype=int),
            ])

            numeric_cols = X_adv.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = [c for c in X_adv.columns if c not in numeric_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]),
                        numeric_cols,
                    ),
                    (
                        "cat",
                        Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]),
                        categorical_cols,
                    ),
                ],
                remainder="drop",
            )

            model = LogisticRegression(max_iter=300, solver="saga", random_state=42)
            pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
            pipe.fit(X_adv, y_adv)
            proba = np.asarray(pipe.predict_proba(X_adv), dtype=float)[:, 1]
            auc = float(roc_auc_score(y_adv, proba))
            drift_level = "low"
            if auc >= 0.80:
                drift_level = "high"
            elif auc >= 0.65:
                drift_level = "moderate"
            return {
                "enabled": True,
                "auc": auc,
                "drift_level": drift_level,
                "train_rows_used": int(len(train_X)),
                "test_rows_used": int(len(test_X)),
            }
        except Exception as exc:
            return {"enabled": False, "reason": f"error:{exc}"}

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        file_path = state.get("file_path")
        target_column = state.get("target_column")
        if not file_path:
            raise ValueError("file_path missing in state")

        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Dataset is empty")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        rows, cols = df.shape
        missing_pct = float(df.isna().sum().sum() / (rows * cols) * 100) if rows else 0.0
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        identifier_columns = self._detect_identifier_columns(df, target_column)
        relationship_insights = self._compute_relationship_insights(df, target_column, numeric_cols)

        target_series = df[target_column].dropna()
        target_min = None
        target_max = None
        target_is_non_negative = False
        if not target_series.empty and pd.api.types.is_numeric_dtype(target_series):
            target_min = float(target_series.min())
            target_max = float(target_series.max())
            target_is_non_negative = target_min >= 0.0

        adversarial_validation: Dict[str, Any] = {"enabled": False, "reason": "test_file_not_provided"}
        test_file_path = str(state.get("test_file_path") or "").strip()
        if test_file_path:
            try:
                test_df = pd.read_csv(test_file_path)
                if not test_df.empty:
                    adversarial_validation = self._compute_adversarial_validation_auc(df, test_df, target_column)
            except Exception as exc:
                adversarial_validation = {"enabled": False, "reason": f"test_load_error:{exc}"}

        state["raw_df"] = df
        state["dataset_summary"] = {
            "rows": rows,
            "cols": cols,
            "missing_pct": missing_pct,
            "dtypes": dtypes,
            "missing_by_column": df.isna().sum().to_dict(),
            "identifier_columns": identifier_columns,
            "relationship_insights": relationship_insights,
            "target_min": target_min,
            "target_max": target_max,
            "target_is_non_negative": target_is_non_negative,
            "adversarial_validation": adversarial_validation,
        }
        state.setdefault("reasoning_trace", []).append(
            f"Loaded dataset with {rows} rows, {cols} columns, missing={missing_pct:.2f}%"
        )
        if identifier_columns:
            state.setdefault("reasoning_trace", []).append(
                f"Detected identifier-like columns: {', '.join(identifier_columns)}"
            )
        if adversarial_validation.get("enabled"):
            auc = float(adversarial_validation.get("auc", 0.5))
            state.setdefault("reasoning_trace", []).append(
                f"Adversarial validation AUC={auc:.4f} (drift={adversarial_validation.get('drift_level', 'unknown')})"
            )
        return state
