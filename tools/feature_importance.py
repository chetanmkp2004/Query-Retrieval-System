"""Feature importance extraction tool."""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Type

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class StateInput(BaseModel):
    state: Dict[str, Any] = Field(..., description="Current mutable agent state")


class FeatureImportanceTool(BaseTool):
    name: str = "feature_importance"
    description: str = "Extract top 5 feature importances for tree-based best models"
    args_schema: ClassVar[Type[BaseModel]] = StateInput

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        model = state.get("best_model_object")
        feature_names: List[str] = state.get("transformed_feature_names") or []
        if model is None:
            return state

        if not hasattr(model, "feature_importances_"):
            state.setdefault("reasoning_trace", []).append(
                "Best model is not tree-based; feature importance not available"
            )
            return state

        importances = model.feature_importances_
        if len(importances) == 0:
            return state

        indices = np.argsort(importances)[::-1][:5]
        top_features = []
        for idx in indices:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            top_features.append({"feature": name, "importance": float(importances[idx])})

        state["final_feature_importance"] = top_features
        state.setdefault("reasoning_trace", []).append(
            "Extracted top 5 feature importances from best tree-based model"
        )
        return state
