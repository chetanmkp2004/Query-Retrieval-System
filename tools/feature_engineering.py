"""Feature engineering tool."""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Type

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class StateInput(BaseModel):
    state: Dict[str, Any] = Field(..., description="Current mutable agent state")


class FeatureEngineeringTool(BaseTool):
    name: str = "feature_engineering"
    description: str = "Create lightweight engineered columns from tabular data"
    args_schema: ClassVar[Type[BaseModel]] = StateInput
    MAX_NEW_FEATURES: int = 8

    @staticmethod
    def _safe_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    @staticmethod
    def _extract_top_pairs(summary: Dict[str, Any], numeric_cols: List[str]) -> List[tuple[str, str]]:
        raw_pairs = summary.get("relationship_insights", {}).get("top_numeric_correlations", [])
        parsed_pairs: List[tuple[str, str]] = []
        for row in raw_pairs:
            pair_text = str(row.get("pair", "")).strip()
            if "~" not in pair_text:
                continue
            left, right = [part.strip() for part in pair_text.split("~", 1)]
            if left in numeric_cols and right in numeric_cols and left != right:
                normalized = tuple(sorted((left, right)))
                if normalized not in parsed_pairs:
                    parsed_pairs.append(normalized)
        return parsed_pairs

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = state.get("raw_df")
        target_column = state.get("target_column")
        if df is None:
            raise ValueError("raw_df missing in state")
        if not target_column:
            raise ValueError("target_column missing in state")

        engineered_df = df.copy()
        summary = state.get("dataset_summary", {})
        identifier_columns = [
            col for col in summary.get("identifier_columns", []) if col in engineered_df.columns
        ]

        base_numeric_cols = [
            col
            for col in engineered_df.select_dtypes(include=["number"]).columns.tolist()
            if col != target_column and col not in identifier_columns
        ]
        top_pairs = self._extract_top_pairs(summary=summary, numeric_cols=base_numeric_cols)
        if not top_pairs and len(base_numeric_cols) >= 2:
            fallback_pairs = []
            for i in range(len(base_numeric_cols)):
                for j in range(i + 1, len(base_numeric_cols)):
                    fallback_pairs.append((base_numeric_cols[i], base_numeric_cols[j]))
                    if len(fallback_pairs) >= 3:
                        break
                if len(fallback_pairs) >= 3:
                    break
            top_pairs = fallback_pairs

        created_columns: List[str] = []
        used_base_columns: List[str] = []
        interaction_pairs_used: List[str] = []

        for left, right in top_pairs:
            if len(created_columns) >= self.MAX_NEW_FEATURES:
                break
            left_num = self._safe_numeric(engineered_df[left])
            right_num = self._safe_numeric(engineered_df[right])

            interaction_col = f"{left}_x_{right}"
            if interaction_col not in engineered_df.columns:
                engineered_df[interaction_col] = left_num * right_num
                created_columns.append(interaction_col)
                interaction_pairs_used.append(f"{left} x {right}")
                if left not in used_base_columns:
                    used_base_columns.append(left)
                if right not in used_base_columns:
                    used_base_columns.append(right)
            if len(created_columns) >= self.MAX_NEW_FEATURES:
                break

            ratio_col = f"{left}_div_{right}"
            if ratio_col not in engineered_df.columns:
                safe_denominator = right_num.replace(0, np.nan)
                engineered_df[ratio_col] = left_num / safe_denominator
                created_columns.append(ratio_col)
            if len(created_columns) >= self.MAX_NEW_FEATURES:
                break

        if not used_base_columns:
            used_base_columns = base_numeric_cols[:2]

        state["raw_df"] = engineered_df
        summary["engineered_feature_columns"] = created_columns
        summary["engineered_feature_count"] = len(created_columns)
        summary["cols"] = int(engineered_df.shape[1])
        state["dataset_summary"] = summary
        state["engineered_feature_columns"] = created_columns
        state["engineered_feature_count"] = len(created_columns)
        state["engineered_base_columns"] = used_base_columns
        state["engineered_aggregate_base_columns"] = []
        state["features_added_this_round"] = len(created_columns) > 0
        state["engineered_feature_metadata"] = {
            "max_new_features": self.MAX_NEW_FEATURES,
            "pair_candidates": [f"{l} x {r}" for l, r in top_pairs],
            "pairs_used": interaction_pairs_used,
            "created_columns": created_columns,
        }

        print(f"[FE Tool] Created {len(created_columns)} engineered columns from base columns: {used_base_columns}")
        print(f"[FE Tool] Engineered columns: {created_columns}")
        
        if created_columns:
            state.setdefault("reasoning_trace", []).append(
                "Feature engineering complete: "
                f"added {len(created_columns)} columns ({', '.join(created_columns[:8])}"
                f"{'...' if len(created_columns) > 8 else ''})"
            )
        else:
            state.setdefault("reasoning_trace", []).append(
                "Feature engineering complete: no additional columns were created"
            )

        return state
