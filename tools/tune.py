"""Hyperparameter tuning tool."""
from __future__ import annotations

from importlib import import_module
from inspect import signature
from typing import Any, ClassVar, Dict, Type
import logging
import warnings
import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import all_estimators


logger = logging.getLogger(__name__)


class StateInput(BaseModel):
    state: Dict[str, Any] = Field(..., description="Current mutable agent state")


class HyperparameterTuningTool(BaseTool):
    name: str = "tune_model"
    description: str = "Tune the current best model using constrained grid search"
    args_schema: ClassVar[Type[BaseModel]] = StateInput

    @staticmethod
    def _base_model_name(model_name: str) -> str:
        return model_name.replace(" (tuned)", "").strip()

    @staticmethod
    def _resolve_estimator_class(model_name: str, problem_type: str):
        type_filter = "regressor" if problem_type == "regression" else "classifier"
        available = {name.lower(): cls for name, cls in all_estimators(type_filter=type_filter)}
        aliases = {
            "histogramgradientboostingregressor": "histgradientboostingregressor",
            "histogramgradientboostingclassifier": "histgradientboostingclassifier",
        }

        if problem_type == "regression":
            external_paths = {
                "xgbregressor": ("xgboost", "XGBRegressor"),
                "lgbmregressor": ("lightgbm", "LGBMRegressor"),
                "catboostregressor": ("catboost", "CatBoostRegressor"),
            }
        else:
            external_paths = {
                "xgbclassifier": ("xgboost", "XGBClassifier"),
                "lgbmclassifier": ("lightgbm", "LGBMClassifier"),
                "catboostclassifier": ("catboost", "CatBoostClassifier"),
            }

        key = model_name.lower()
        resolved_key = aliases.get(key, key)
        estimator_cls = available.get(resolved_key)
        if estimator_cls is not None:
            return estimator_cls

        if resolved_key in external_paths:
            module_name, class_name = external_paths[resolved_key]
            try:
                module = import_module(module_name)
                return getattr(module, class_name, None)
            except Exception:
                return None

        return None

    @staticmethod
    def _instantiate_estimator(estimator_cls):
        init_params = signature(estimator_cls.__init__).parameters
        kwargs: Dict[str, Any] = {}
        estimator_name = getattr(estimator_cls, "__name__", "").lower()
        if "random_state" in init_params:
            kwargs["random_state"] = 42
        if "n_jobs" in init_params and estimator_name != "logisticregression":
            kwargs["n_jobs"] = -1
        if "thread_count" in init_params:
            kwargs["thread_count"] = -1
        if "max_iter" in init_params:
            kwargs["max_iter"] = 500
        if "n_estimators" in init_params:
            kwargs["n_estimators"] = 500
        if "verbose" in init_params:
            kwargs["verbose"] = 0
        if "n_iter_no_change" in init_params:
            kwargs["n_iter_no_change"] = 10
        if "validation_fraction" in init_params:
            kwargs["validation_fraction"] = 0.1
        if estimator_name.startswith("lgbm"):
            kwargs["verbosity"] = -1
            kwargs["verbose"] = -1
            kwargs["force_row_wise"] = True
        return estimator_cls(**kwargs)

    @staticmethod
    def _grid_combinations(param_grid: Dict[str, Any]) -> int:
        combos = 1
        for values in param_grid.values():
            if isinstance(values, list):
                combos *= max(1, len(values))
            else:
                combos *= 1
        return int(combos)

    @staticmethod
    def _cap_grid_combinations(param_grid: Dict[str, Any], max_combinations: int) -> Dict[str, Any]:
        capped: Dict[str, Any] = {k: (list(v) if isinstance(v, list) else [v]) for k, v in param_grid.items()}
        if max_combinations <= 0:
            return capped

        while HyperparameterTuningTool._grid_combinations(capped) > max_combinations:
            reducible = [key for key, values in capped.items() if isinstance(values, list) and len(values) > 1]
            if not reducible:
                break
            key_to_reduce = max(reducible, key=lambda k: len(capped[k]))
            capped[key_to_reduce] = capped[key_to_reduce][:-1]

        return capped

    @staticmethod
    def _sanitize_param_grid(param_grid: Dict[str, Any], estimator: Any) -> Dict[str, Any]:
        valid_params = set(estimator.get_params().keys()) if hasattr(estimator, "get_params") else set()
        sanitized: Dict[str, Any] = {}
        for key, value in (param_grid or {}).items():
            target_key = key
            if target_key not in valid_params and f"model__{key}" in valid_params:
                target_key = f"model__{key}"
            if target_key not in valid_params:
                continue
            if isinstance(value, (list, tuple)):
                values = [v for v in value if v is not None]
            else:
                values = [value]
            if values:
                sanitized[target_key] = values

        estimator_name = estimator.__class__.__name__.lower()
        if estimator_name == "logisticregression":
            sanitized.pop("penalty", None)
            sanitized.pop("l1_ratio", None)
            sanitized.pop("model__penalty", None)
            sanitized.pop("model__l1_ratio", None)
        return sanitized

    @staticmethod
    def _default_tuning_grid(base_model_name: str, problem_type: str) -> Dict[str, Any]:
        model_key = base_model_name.lower()
        if problem_type == "regression":
            defaults: Dict[str, Dict[str, Any]] = {
                "linearregression": {
                    "fit_intercept": [True, False],
                },
                "randomforestregressor": {
                    "n_estimators": [200, 400, 600],
                    "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 2, 4],
                },
                "gradientboostingregressor": {
                    "n_estimators": [150, 250, 400],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [2, 3, 4],
                    "subsample": [0.8, 1.0],
                },
                "histgradientboostingregressor": {
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [None, 6, 10],
                    "max_leaf_nodes": [31, 63],
                    "min_samples_leaf": [20, 30],
                    "l2_regularization": [0.0, 0.1],
                },
                "lgbmregressor": {
                    "learning_rate": [0.03, 0.05, 0.1],
                    "n_estimators": [300, 500],
                    "num_leaves": [31, 63],
                    "min_child_samples": [20, 30],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_alpha": [0.0, 0.05],
                    "reg_lambda": [0.0, 0.05],
                },
                "xgbregressor": {
                    "learning_rate": [0.03, 0.05, 0.1],
                    "n_estimators": [300, 500],
                    "max_depth": [4, 6],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_alpha": [0.0, 0.05],
                    "reg_lambda": [1.0, 2.0],
                },
            }
            return defaults.get(model_key, {})
        defaults_cls: Dict[str, Dict[str, Any]] = {
            "logisticregression": {
                "C": [0.3, 1.0, 3.0, 10.0],
                "solver": ["lbfgs", "saga"],
                "max_iter": [400, 700],
            },
            "lgbmclassifier": {
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [300, 500, 700],
                "num_leaves": [31, 63, 127],
                "min_child_samples": [20, 40, 80],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_alpha": [0.0, 0.05, 0.1],
                "reg_lambda": [0.0, 0.05, 0.1],
            },
            "xgbclassifier": {
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [300, 500, 700],
                "max_depth": [4, 6, 8],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_alpha": [0.0, 0.05, 0.1],
                "reg_lambda": [1.0, 2.0, 4.0],
            },
            "catboostclassifier": {
                "depth": [6, 8, 10],
                "learning_rate": [0.03, 0.05, 0.1],
                "iterations": [300, 500, 700],
                "l2_leaf_reg": [3, 5, 7],
            },
            "randomforestclassifier": {
                "n_estimators": [200, 400, 600],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 4],
            },
            "gradientboostingclassifier": {
                "n_estimators": [150, 250, 400],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
            },
            "histgradientboostingclassifier": {
                "learning_rate": [0.03, 0.1],
                "max_depth": [None, 8],
                "max_leaf_nodes": [31, 63],
                "min_samples_leaf": [20, 40],
                "l2_regularization": [0.0, 0.1],
            }
        }
        return defaults_cls.get(model_key, {})

    @staticmethod
    def _safe_neg_rmsle_scorer(estimator, X, y):
        y_true = np.asarray(y, dtype=float)
        y_true = np.clip(y_true, 0.0, None)
        y_pred = estimator.predict(X)
        y_pred = np.asarray(y_pred, dtype=float)
        y_pred = np.clip(y_pred, 0.0, None)
        rmsle = float(np.sqrt(mean_squared_log_error(y_true, y_pred)))
        return -rmsle

    @staticmethod
    def _safe_neg_mse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return -float(mean_squared_error(y_true, y_pred))

    @staticmethod
    def _safe_neg_mae(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return -float(mean_absolute_error(y_true, y_pred))

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        tuning_pass_count = int(state.get("tuning_pass_count", 0))
        max_tuning_passes = int(state.get("max_tuning_passes", 3) or 3)
        if tuning_pass_count >= max_tuning_passes:
            state.setdefault("reasoning_trace", []).append(
                f"Tuning skipped because pass budget is exhausted ({tuning_pass_count}/{max_tuning_passes})"
            )
            return state

        best_model_name = state.get("best_model_name")
        X_train = state.get("X_train")
        X_train_raw = state.get("X_train_raw")
        y_train = state.get("y_train")
        if not best_model_name:
            raise ValueError("best_model_name is missing for tuning")

        selected_model_name = str(best_model_name)

        if selected_model_name.startswith("SoftVotingEnsembleClassifier"):
            cv_scores = state.get("cv_scores", {}) or {}
            current_metrics = state.get("current_metrics", {}) or {}
            trained_models = state.get("trained_models", {}) or {}
            candidates: list[tuple[str, float]] = []
            for model_name, stats in cv_scores.items():
                name = str(model_name)
                lowered = name.lower()
                if lowered.startswith("softvotingensembleclassifier") or lowered.startswith("ridgestackingensemble"):
                    continue
                cv_mean = float((stats or {}).get("cv_mean", float("-inf")))
                candidates.append((name, cv_mean))

            if not candidates:
                for model_name, metric_blob in current_metrics.items():
                    name = str(model_name)
                    lowered = name.lower()
                    if lowered.startswith("softvotingensembleclassifier") or lowered.startswith("ridgestackingensemble"):
                        continue
                    metric_map = metric_blob or {}
                    if "CV_Accuracy" in metric_map:
                        score = float(metric_map.get("CV_Accuracy", float("-inf")))
                    elif "CV_MSE" in metric_map:
                        score = -float(metric_map.get("CV_MSE", float("inf")))
                    elif "CV_MAE" in metric_map:
                        score = -float(metric_map.get("CV_MAE", float("inf")))
                    elif "CV_RMSLE" in metric_map:
                        score = -float(metric_map.get("CV_RMSLE", float("inf")))
                    else:
                        score = float("-inf")
                    candidates.append((name, score))

            if not candidates:
                for model_name in trained_models.keys():
                    name = str(model_name)
                    lowered = name.lower()
                    if lowered.startswith("softvotingensembleclassifier") or lowered.startswith("ridgestackingensemble"):
                        continue
                    candidates.append((name, float("-inf")))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                selected_model_name = str(candidates[0][0])
                state.setdefault("reasoning_trace", []).append(
                    f"Best model is soft-voting ensemble; tuning strongest base model {selected_model_name}"
                )
            else:
                state.setdefault("reasoning_trace", []).append("Tuning skipped for soft-voting ensemble (no tunable base model found)")
                state["tuning_performed"] = True
                state["tuning_pass_count"] = tuning_pass_count + 1
                return state

        if str(selected_model_name).startswith("RidgeStackingEnsemble"):
            state.setdefault("reasoning_trace", []).append("Tuning skipped for stacking ensemble")
            state["tuning_performed"] = True
            state["tuning_pass_count"] = tuning_pass_count + 1
            return state

        if str(selected_model_name).startswith("RankRidgeStackingClassifier"):
            state.setdefault("reasoning_trace", []).append("Tuning skipped for rank-ridge stacking ensemble")
            state["tuning_performed"] = True
            state["tuning_pass_count"] = tuning_pass_count + 1
            return state

        if y_train is None:
            raise ValueError("Training matrices are missing for tuning")

        model_in_state = state.get("best_model_object") if str(selected_model_name) == str(best_model_name) else (state.get("trained_models", {}) or {}).get(str(selected_model_name))
        if model_in_state is not None:
            estimator = clone(model_in_state)
            is_pipeline_like = hasattr(estimator, "named_steps")
        else:
            estimator = None
            is_pipeline_like = False

        X_tune = X_train_raw if is_pipeline_like and X_train_raw is not None else X_train
        if X_tune is None:
            raise ValueError("No training features available for tuning")

        train_rows = int(getattr(X_tune, "shape", [len(y_train), 1])[0])
        if train_rows < 1000:
            state.setdefault("reasoning_trace", []).append(
                f"Tuning skipped: dataset too small for robust tuning ({train_rows} rows < 1000)"
            )
            state["tuning_performed"] = True
            state["tuning_pass_count"] = tuning_pass_count + 1
            return state

        base_model_name = self._base_model_name(str(selected_model_name))
        base_model_key = base_model_name.lower()

        tuned_model_types = [str(v).lower() for v in state.get("tuned_model_types", [])]
        if base_model_key in tuned_model_types:
            state.setdefault("reasoning_trace", []).append(
                f"Tuning skipped for {base_model_name}: already tuned for this model type"
            )
            state["tuning_performed"] = True
            state["tuning_pass_count"] = tuning_pass_count + 1
            return state

        param_grid: Dict[str, Any] = {}
        llm_grids = state.get("suggested_hyperparams", {}) or {}
        if estimator is None:
            estimator_cls = self._resolve_estimator_class(
                model_name=base_model_name,
                problem_type=str(state.get("problem_type")),
            )
            if estimator_cls is not None:
                estimator = self._instantiate_estimator(estimator_cls)

        if base_model_name in llm_grids and isinstance(llm_grids[base_model_name], dict):
            llm_grid = llm_grids[base_model_name]
            normalized_grid: Dict[str, Any] = {}
            for key, value in llm_grid.items():
                if isinstance(value, (list, tuple)):
                    normalized_grid[key] = list(value)
                else:
                    normalized_grid[key] = [value]
            param_grid = normalized_grid

        if estimator is None:
            state.setdefault("reasoning_trace", []).append(
                f"No tuning space configured for {base_model_name}; skipping tuning"
            )
            state["tuning_performed"] = True
            state["tuning_pass_count"] = tuning_pass_count + 1
            return state

        param_grid = self._sanitize_param_grid(param_grid, estimator)
        fallback_grid = self._default_tuning_grid(base_model_name, str(state.get("problem_type")))
        fallback_grid = self._sanitize_param_grid(fallback_grid, estimator)

        if (
            str(state.get("problem_type")) == "classification"
            and train_rows >= 250000
            and param_grid
            and fallback_grid
        ):
            current_combo_count = self._grid_combinations(param_grid)
            if current_combo_count < 24:
                augmented = {k: list(v) for k, v in param_grid.items()}
                for key, values in fallback_grid.items():
                    if key not in augmented:
                        augmented[key] = list(values)
                param_grid = augmented
                state.setdefault("reasoning_trace", []).append(
                    f"Augmented LLM tuning grid for {base_model_name} to widen large-scale search space"
                )

        if not param_grid:
            if fallback_grid:
                param_grid = fallback_grid
                state.setdefault("reasoning_trace", []).append(
                    f"Using fallback tuning grid for {base_model_name}"
                )
            else:
                state.setdefault("reasoning_trace", []).append(
                    f"No valid hyperparameter grid available for {base_model_name}; skipping tuning"
                )
                state["tuning_performed"] = True
                state["tuning_pass_count"] = tuning_pass_count + 1
                return state

        max_grid_combinations = 48
        param_grid = self._cap_grid_combinations(param_grid, max_combinations=max_grid_combinations)
        combo_count = self._grid_combinations(param_grid)

        metric_objective = str(state.get("metric_objective", "")).strip().lower()
        if state.get("problem_type") == "regression":
            if metric_objective == "rmsle":
                scoring = self._safe_neg_rmsle_scorer
            elif metric_objective == "mae":
                scoring = make_scorer(self._safe_neg_mae, greater_is_better=True)
            else:
                scoring = make_scorer(self._safe_neg_mse, greater_is_better=True)
        else:
            scoring = "accuracy"

        cv_splits = [
            (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
            for tr, va in (state.get("cv_split_indices") or [])
        ]
        effective_cv = cv_splits

        if train_rows >= 250000 and str(state.get("problem_type")) == "classification":
            n_iter = min(20, max(10, combo_count))
        elif train_rows >= 250000:
            n_iter = min(12, max(6, combo_count))
        else:
            n_iter = min(14, max(6, combo_count))

        cv_count = len(effective_cv) if effective_cv else 5
        logger.info(
            "Tuning %s with randomized search (candidate_combinations=%d, n_iter=%d, cv=%d)",
            base_model_name,
            combo_count,
            n_iter,
            cv_count,
        )
        state.setdefault("reasoning_trace", []).append(
            f"Tuning {base_model_name} with randomized search (n_iter={n_iter}, cv={cv_count})"
        )
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=effective_cv or 5,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            if "lgbm" in str(base_model_name).lower():
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    search.fit(X_tune, y_train)
            else:
                search.fit(X_tune, y_train)

        tuned_cv = float(getattr(search, "best_score_", float("-inf")))
        baseline_cv = float((state.get("cv_scores", {}) or {}).get(base_model_name, {}).get("cv_mean", float("-inf")))
        if np.isfinite(baseline_cv) and tuned_cv <= baseline_cv + 1e-6:
            state["tuning_performed"] = True
            state["tuning_pass_count"] = tuning_pass_count + 1
            state["tuned_model_types"] = [*state.get("tuned_model_types", []), base_model_key]
            state.setdefault("reasoning_trace", []).append(
                f"Rejected tuned {base_model_name}: tuned_cv={tuned_cv:.6f} <= baseline_cv={baseline_cv:.6f}; keeping baseline model"
            )
            return state

        state["best_model_object"] = search.best_estimator_
        state["best_model_name"] = base_model_name + " (tuned)"
        state.setdefault("trained_models", {})[state["best_model_name"]] = search.best_estimator_
        if state["best_model_name"] not in state.setdefault("models_tried", []):
            state["models_tried"].append(state["best_model_name"])
        state.setdefault("model_params", {})[state["best_model_name"]] = search.best_params_
        state["tuning_performed"] = True
        state["tuning_pass_count"] = tuning_pass_count + 1
        state["tuned_model_types"] = [*state.get("tuned_model_types", []), base_model_key]
        state.setdefault("reasoning_trace", []).append(
            f"Tuned model {base_model_name}; best params: {search.best_params_}"
        )
        return state
