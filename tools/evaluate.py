"""Model evaluation tool."""
from __future__ import annotations

import logging
import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, ClassVar, Dict, Type
import warnings

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None


logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)


class StateInput(BaseModel):
    state: Dict[str, Any] = Field(..., description="Current mutable agent state")


class RidgeStackingEnsembleRegressor:
    def __init__(self, base_models: Dict[str, Any], meta_model: Ridge, target_transformed: bool):
        self.base_models = base_models
        self.meta_model = meta_model
        self.target_transformed = target_transformed

    def predict(self, X):
        preds = []
        for model in self.base_models.values():
            pred = np.asarray(model.predict(X), dtype=float)
            if self.target_transformed:
                pred = np.expm1(pred)
            preds.append(pred)
        meta_X = np.column_stack(preds)
        return self.meta_model.predict(meta_X)


class SoftVotingEnsembleClassifier:
    def __init__(self, base_models: Dict[str, Any], classes_: np.ndarray, weights: Dict[str, float], threshold: float = 0.5, positive_class: Any = None, platt_a: float | None = None, platt_b: float | None = None):
        self.base_models = base_models
        self.classes_ = np.asarray(classes_)
        self.weights = {k: float(v) for k, v in weights.items()}
        self.threshold = float(threshold)
        self.positive_class = positive_class
        self.platt_a = platt_a
        self.platt_b = platt_b

    @staticmethod
    def _to_proba(model: Any, X: Any, classes_: np.ndarray) -> np.ndarray:
        n_classes = len(classes_)
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X), dtype=float)
            model_classes = np.asarray(getattr(model, "classes_", classes_))
            aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
            for class_idx, class_value in enumerate(classes_):
                hits = np.where(model_classes == class_value)[0]
                if len(hits) > 0:
                    aligned[:, class_idx] = proba[:, int(hits[0])]
            row_sum = aligned.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return aligned / row_sum

        preds = np.asarray(model.predict(X))
        aligned = np.zeros((len(preds), n_classes), dtype=float)
        for class_idx, class_value in enumerate(classes_):
            aligned[:, class_idx] = (preds == class_value).astype(float)
        return aligned

    def predict_proba(self, X: Any) -> np.ndarray:
        weighted = None
        weight_sum = 0.0
        for name, model in self.base_models.items():
            w = float(self.weights.get(name, 1.0))
            proba = self._to_proba(model, X, self.classes_)
            weighted = proba * w if weighted is None else weighted + (proba * w)
            weight_sum += w
        if weighted is None:
            return np.zeros((0, len(self.classes_)), dtype=float)
        if weight_sum <= 0.0:
            weight_sum = 1.0
        proba = weighted / weight_sum
        if len(self.classes_) == 2 and self.positive_class is not None and self.platt_a is not None and self.platt_b is not None:
            hit = np.where(self.classes_ == self.positive_class)[0]
            if len(hit) > 0:
                pos_idx = int(hit[0])
                raw_pos = np.clip(proba[:, pos_idx], 1e-6, 1.0 - 1e-6)
                calibrated_pos = 1.0 / (1.0 + np.exp(-(float(self.platt_a) * raw_pos + float(self.platt_b))))
                calibrated_pos = np.clip(calibrated_pos, 0.0, 1.0)
                neg_idx = 1 - pos_idx
                proba[:, pos_idx] = calibrated_pos
                proba[:, neg_idx] = 1.0 - calibrated_pos
        return proba

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.size == 0:
            return np.array([], dtype=self.classes_.dtype)
        if len(self.classes_) == 2 and self.positive_class is not None:
            hit = np.where(self.classes_ == self.positive_class)[0]
            if len(hit) > 0:
                pos_idx = int(hit[0])
                neg_label = self.classes_[0] if self.classes_[0] != self.positive_class else self.classes_[1]
                return np.where(proba[:, pos_idx] >= self.threshold, self.positive_class, neg_label)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class RankRidgeStackingClassifier:
    def __init__(self, base_models: Dict[str, Any], classes_: np.ndarray, model_order: list[str], ridge_model: Ridge, threshold: float = 0.5, positive_class: Any = None):
        self.base_models = base_models
        self.classes_ = np.asarray(classes_)
        self.model_order = list(model_order)
        self.ridge_model = ridge_model
        self.threshold = float(threshold)
        self.positive_class = positive_class

    @staticmethod
    def _to_pos_proba(model: Any, X: Any, positive_class: Any) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            preds = np.asarray(model.predict(X), dtype=object)
            return (preds == positive_class).astype(float)
        proba = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(model, "classes_", []))
        hit = np.where(classes == positive_class)[0]
        if len(hit) == 0 or proba.ndim != 2 or proba.shape[1] <= int(hit[0]):
            preds = np.asarray(model.predict(X), dtype=object)
            return (preds == positive_class).astype(float)
        return np.asarray(proba[:, int(hit[0])], dtype=float)

    @staticmethod
    def _rank_percentile(values: np.ndarray) -> np.ndarray:
        ranked = pd.Series(values).rank(method="average", pct=True)
        arr = np.asarray(ranked, dtype=float).copy()
        arr[~np.isfinite(arr)] = 0.5
        return arr

    def _build_meta_features(self, X: Any) -> np.ndarray:
        if self.positive_class is None:
            return np.zeros((len(X), 0), dtype=float)
        cols = []
        for name in self.model_order:
            model = self.base_models.get(name)
            if model is None:
                continue
            pos = self._to_pos_proba(model, X, self.positive_class)
            cols.append(self._rank_percentile(pos))
        if not cols:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(cols)

    def predict_proba(self, X: Any) -> np.ndarray:
        n_classes = len(self.classes_)
        meta_X = self._build_meta_features(X)
        if meta_X.shape[1] == 0 or n_classes != 2 or self.positive_class is None:
            uniform = np.full((len(X), max(1, n_classes)), 1.0 / max(1, n_classes), dtype=float)
            return uniform

        raw = np.asarray(self.ridge_model.predict(meta_X), dtype=float)
        pos = np.clip(raw, 0.0, 1.0)
        hit = np.where(self.classes_ == self.positive_class)[0]
        if len(hit) == 0:
            uniform = np.full((len(X), n_classes), 1.0 / n_classes, dtype=float)
            return uniform
        pos_idx = int(hit[0])
        neg_idx = 1 - pos_idx
        proba = np.zeros((len(pos), n_classes), dtype=float)
        proba[:, pos_idx] = pos
        proba[:, neg_idx] = 1.0 - pos
        return proba

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.size == 0:
            return np.array([], dtype=self.classes_.dtype)
        if len(self.classes_) == 2 and self.positive_class is not None:
            hit = np.where(self.classes_ == self.positive_class)[0]
            if len(hit) > 0:
                pos_idx = int(hit[0])
                neg_label = self.classes_[0] if self.classes_[0] != self.positive_class else self.classes_[1]
                return np.where(proba[:, pos_idx] >= self.threshold, self.positive_class, neg_label)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class EvaluateModelTool(BaseTool):
    name: str = "evaluate_models"
    description: str = "Evaluate trained models and select best one"
    args_schema: ClassVar[Type[BaseModel]] = StateInput

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

    @staticmethod
    def _objective_label(objective: str) -> str:
        obj = objective.lower()
        if obj == "rmsle":
            return "RMSLE"
        if obj == "mae":
            return "MAE"
        return "MSE"

    @staticmethod
    def _convert_score_to_metric(objective: str, score: float) -> float:
        return -float(score)

    @staticmethod
    def _inverse_if_transformed(preds: np.ndarray, transformed: bool) -> np.ndarray:
        if transformed:
            return np.expm1(preds)
        return preds

    @staticmethod
    def _stability_adjusted_score(mean_score: float, std_score: float, problem_type: str) -> float:
        if problem_type == "classification":
            return float(mean_score - (0.35 * std_score))
        return float(mean_score - (0.25 * std_score))

    @staticmethod
    def _choose_positive_class(classes_: np.ndarray):
        values = [str(v).strip().lower() for v in classes_]
        positive_tokens = {"presence", "present", "yes", "y", "true", "positive", "1", "disease", "abnormal"}
        for original, lowered in zip(classes_, values):
            if lowered in positive_tokens:
                return original
        try:
            sorted_classes = sorted(list(classes_))
            return sorted_classes[-1]
        except Exception:
            return classes_[-1]

    @staticmethod
    def _optimize_binary_threshold(y_true: np.ndarray, pos_proba: np.ndarray, positive_class: Any) -> tuple[float, float]:
        y_true_arr = np.asarray(y_true)
        proba = np.asarray(pos_proba, dtype=float)
        valid_mask = np.isfinite(proba)
        if valid_mask.sum() == 0:
            return 0.5, -np.inf

        y_eval = y_true_arr[valid_mask]
        p_eval = proba[valid_mask]
        thresholds = np.arange(0.30, 0.701, 0.02)

        classes = pd.Series(y_eval).dropna().unique().tolist()
        if len(classes) != 2 or positive_class not in classes:
            return 0.5, -np.inf
        negative_class = classes[0] if classes[0] != positive_class else classes[1]

        best_threshold = 0.5
        best_accuracy = -np.inf
        for t in thresholds:
            preds = np.where(p_eval >= t, positive_class, negative_class)
            acc = float(accuracy_score(y_eval, preds))
            if acc > best_accuracy + 1e-12:
                best_accuracy = acc
                best_threshold = float(t)

        return best_threshold, best_accuracy

    @staticmethod
    def _build_corr_matrix(oof_preds_by_model: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        names = list(oof_preds_by_model.keys())
        if len(names) < 2:
            return {name: {name: 1.0} for name in names}
        matrix = np.corrcoef(np.vstack([oof_preds_by_model[name] for name in names]))
        corr_dict: Dict[str, Dict[str, float]] = {}
        for i, left in enumerate(names):
            corr_dict[left] = {}
            for j, right in enumerate(names):
                value = float(matrix[i, j]) if not np.isnan(matrix[i, j]) else 0.0
                corr_dict[left][right] = value
        return corr_dict

    def _evaluate_rank_ridge_binary(
        self,
        y_true: np.ndarray,
        y_proba_by_model: Dict[str, np.ndarray],
        model_order: list[str],
        classes_: np.ndarray,
        positive_class: Any,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any] | None:
        if len(model_order) < 2:
            return None
        pos_hit = np.where(classes_ == positive_class)[0]
        if len(pos_hit) == 0:
            return None
        pos_idx = int(pos_hit[0])

        feature_cols: list[np.ndarray] = []
        used_models: list[str] = []
        for name in model_order:
            proba = np.asarray(y_proba_by_model.get(name, np.array([])), dtype=float)
            if proba.ndim != 2 or proba.shape[0] != len(y_true) or proba.shape[1] <= pos_idx:
                continue
            ranked = pd.Series(np.asarray(proba[:, pos_idx], dtype=float)).rank(method="average", pct=True)
            ranked_arr = np.asarray(ranked, dtype=float).copy()
            ranked_arr[~np.isfinite(ranked_arr)] = 0.5
            feature_cols.append(ranked_arr)
            used_models.append(name)

        if len(used_models) < 2:
            return None

        X_rank = np.column_stack(feature_cols)
        y_bin = (np.asarray(y_true) == positive_class).astype(int)

        alpha_grid = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        k_grid_raw = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 36, len(used_models)]
        k_grid = sorted({k for k in k_grid_raw if 2 <= k <= len(used_models)})

        best: Dict[str, Any] | None = None
        neg_label = classes_[0] if classes_[0] != positive_class else classes_[1]
        for alpha in alpha_grid:
            for top_k in k_grid:
                oof_pos = np.zeros(len(y_bin), dtype=float)
                fold_acc = []
                for tr, va in cv_splits:
                    meta = Ridge(alpha=float(alpha), random_state=42)
                    meta.fit(X_rank[tr, :top_k], y_bin[tr])
                    pred_pos = np.clip(np.asarray(meta.predict(X_rank[va, :top_k]), dtype=float), 0.0, 1.0)
                    oof_pos[va] = pred_pos
                    base_pred = (pred_pos >= 0.5).astype(int)
                    fold_acc.append(float(accuracy_score(y_bin[va], base_pred)))

                threshold, threshold_acc = self._optimize_binary_threshold(
                    y_true=np.asarray(y_true),
                    pos_proba=oof_pos,
                    positive_class=positive_class,
                )
                if np.isfinite(threshold_acc):
                    score = float(threshold_acc)
                else:
                    score = float(accuracy_score(y_bin, (oof_pos >= 0.5).astype(int)))
                    threshold = 0.5

                fold_scores_thresh = []
                for _, va in cv_splits:
                    preds = np.where(oof_pos[va] >= threshold, positive_class, neg_label)
                    fold_scores_thresh.append(float(accuracy_score(np.asarray(y_true)[va], preds)))
                score_std = float(np.std(np.asarray(fold_scores_thresh, dtype=float)))

                if best is None or score > float(best["score"]) + 1e-12 or (
                    abs(score - float(best["score"])) <= 1e-12 and score_std < float(best["score_std"]) - 1e-12
                ):
                    best = {
                        "score": score,
                        "score_std": score_std,
                        "threshold": float(threshold),
                        "alpha": float(alpha),
                        "top_k": int(top_k),
                        "used_models": list(used_models[:top_k]),
                        "oof_pos": oof_pos,
                    }

        if best is None:
            return None

        ridge_final = Ridge(alpha=float(best["alpha"]), random_state=42)
        ridge_final.fit(X_rank[:, : int(best["top_k"])], y_bin)
        best["ridge_model"] = ridge_final
        return best

    def _select_diverse_models(
        self,
        model_scores: Dict[str, float],
        corr_matrix: Dict[str, Dict[str, float]],
        threshold: float = 0.95,
    ) -> tuple[list[str], list[str]]:
        ordered = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        kept: list[str] = []
        dropped: list[str] = []
        for name, _ in ordered:
            should_drop = False
            for existing in kept:
                corr_val = abs(float(corr_matrix.get(name, {}).get(existing, 0.0)))
                if corr_val > threshold:
                    should_drop = True
                    break
            if should_drop:
                dropped.append(name)
            else:
                kept.append(name)
        return kept, dropped

    def _optimize_stacking_alpha(
        self,
        X_meta: np.ndarray,
        y_true: np.ndarray,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        objective: str,
    ) -> float:
        def evaluate_alpha(alpha: float) -> float:
            fold_scores = []
            for tr, va in cv_splits:
                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_meta[tr], y_true[tr])
                preds = model.predict(X_meta[va])
                fold_scores.append(self._score_from_objective(objective, y_true[va], preds))
            return float(np.mean(fold_scores))

        if optuna is None:
            grid = [0.01, 0.1, 1.0, 5.0, 10.0]
            scored = [(alpha, evaluate_alpha(alpha)) for alpha in grid]
            return float(max(scored, key=lambda x: x[1])[0])

        study = optuna.create_study(direction="maximize")

        def objective_fn(trial):
            alpha = trial.suggest_float("alpha", 1e-3, 20.0, log=True)
            return evaluate_alpha(alpha)

        study.optimize(objective_fn, n_trials=20, show_progress_bar=False)
        return float(study.best_params["alpha"])

    def _manual_regression_cv_scores(
        self,
        estimator: Any,
        X_train_raw: Any,
        y_train_raw: Any,
        y_train_fit: Any,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        objective: str,
        target_transformed: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores: list[float] = []
        oof_preds = np.zeros(len(y_train_raw), dtype=float)

        y_raw_arr = np.asarray(y_train_raw, dtype=float)
        y_fit_arr = np.asarray(y_train_fit, dtype=float)

        for train_idx, valid_idx in cv_splits:
            pipe = clone(estimator)
            X_tr = X_train_raw.iloc[train_idx]
            X_va = X_train_raw.iloc[valid_idx]
            y_tr_fit = y_fit_arr[train_idx]
            y_va_raw = y_raw_arr[valid_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                pipe.fit(X_tr, y_tr_fit)
            preds = np.asarray(pipe.predict(X_va), dtype=float)
            preds = self._inverse_if_transformed(preds, transformed=target_transformed)

            fold_score = self._score_from_objective(objective, y_va_raw, preds)
            scores.append(fold_score)
            oof_preds[valid_idx] = preds

        return np.asarray(scores, dtype=float), oof_preds

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        trained_models = state.get("trained_models") or {}
        y_train = state.get("y_train")
        X_train_raw = state.get("X_train_raw")
        y_train_raw = state.get("y_train_raw")
        cv_split_indices = state.get("cv_split_indices") or []
        problem_type = state.get("problem_type")
        metric_objective = str(state.get("metric_objective", "")).strip().lower()
        target_transformed = bool(state.get("target_transform_applied", False))

        if not trained_models:
            raise ValueError("No trained models available")
        if y_train is None or X_train_raw is None or y_train_raw is None:
            raise ValueError("Training matrices are missing")
        if not cv_split_indices:
            raise ValueError("Cross-validation split indices are missing")

        cv_splits = [
            (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
            for tr, va in cv_split_indices
        ]
        effective_cv_splits = cv_splits

        metrics: Dict[str, Dict[str, float]] = {}
        overfit_report: Dict[str, Dict[str, float | bool]] = {}
        best_name = ""
        best_model = None
        best_score = -np.inf
        best_adjusted_score = -np.inf
        cv_scores_by_model: Dict[str, Dict[str, float]] = {}
        oof_preds_by_model: Dict[str, np.ndarray] = {}
        oof_label_preds_by_model: Dict[str, np.ndarray] = {}
        oof_pos_proba_by_model: Dict[str, np.ndarray] = {}
        model_score_for_ranking: Dict[str, float] = {}

        if not metric_objective:
            metric_objective = "accuracy" if problem_type == "classification" else "mse"
            state["metric_objective"] = metric_objective

        for name, model in trained_models.items():
            logger.info("Evaluating %s with %d folds", name, len(effective_cv_splits))
            if problem_type == "regression":
                fold_scores, oof_preds = self._manual_regression_cv_scores(
                    estimator=model,
                    X_train_raw=X_train_raw,
                    y_train_raw=y_train_raw,
                    y_train_fit=y_train,
                    cv_splits=effective_cv_splits,
                    objective=metric_objective,
                    target_transformed=target_transformed,
                )
                score = float(np.mean(fold_scores))
                score_std = float(np.std(fold_scores))
                metric_label = self._objective_label(metric_objective)

                train_preds = np.asarray(model.predict(X_train_raw), dtype=float)
                train_preds = self._inverse_if_transformed(train_preds, transformed=target_transformed)
                train_score = self._score_from_objective(
                    metric_objective,
                    np.asarray(y_train_raw, dtype=float),
                    train_preds,
                )

                train_metric = self._convert_score_to_metric(metric_objective, train_score)
                cv_metric = self._convert_score_to_metric(metric_objective, score)
                gap = train_metric - cv_metric
                overfit = gap < -0.01

                metrics[name] = {
                    f"CV_{metric_label}": cv_metric,
                    f"CV_{metric_label}_STD": score_std,
                    f"Train_{metric_label}": train_metric,
                }
                overfit_report[name] = {
                    "train_score": train_metric,
                    "validation_score": cv_metric,
                    "gap": gap,
                    "overfitting": overfit,
                }
                cv_scores_by_model[name] = {
                    "cv_mean": score,
                    "cv_std": score_std,
                    "display_metric": cv_metric,
                }
                oof_preds_by_model[name] = oof_preds
                adjusted = self._stability_adjusted_score(score, score_std, problem_type=str(problem_type))
                model_score_for_ranking[name] = adjusted
            else:
                fold_scores = []
                oof_preds = np.empty(len(y_train_raw), dtype=object)
                y_unique = pd.Series(y_train_raw).dropna().unique()
                binary_mode = len(y_unique) == 2
                positive_class = self._choose_positive_class(np.asarray(y_unique)) if binary_mode else None
                oof_pos_proba = np.full(len(y_train_raw), np.nan, dtype=float)
                for fold_idx, (tr, va) in enumerate(effective_cv_splits, start=1):
                    logger.info("Evaluating %s fold %d/%d", name, fold_idx, len(effective_cv_splits))
                    fold_model = clone(model)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
                        if "lgbm" in str(name).lower():
                            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                                fold_model.fit(X_train_raw.iloc[tr], np.asarray(y_train)[tr])
                        else:
                            fold_model.fit(X_train_raw.iloc[tr], np.asarray(y_train)[tr])
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
                        preds = fold_model.predict(X_train_raw.iloc[va])
                    oof_preds[va] = preds
                    if binary_mode and hasattr(fold_model, "predict_proba"):
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
                                proba = np.asarray(fold_model.predict_proba(X_train_raw.iloc[va]), dtype=float)
                            model_classes = np.asarray(getattr(fold_model, "classes_", y_unique))
                            hit = np.where(model_classes == positive_class)[0]
                            if len(hit) > 0 and proba.ndim == 2 and proba.shape[1] > int(hit[0]):
                                oof_pos_proba[va] = proba[:, int(hit[0])]
                        except Exception:
                            pass
                    fold_scores.append(float(accuracy_score(np.asarray(y_train_raw)[va], preds)))
                fold_scores = np.asarray(fold_scores, dtype=float)
                score = float(np.mean(fold_scores))
                score_std = float(np.std(fold_scores))
                accuracy = score

                train_preds = model.predict(X_train_raw)
                train_accuracy = float(accuracy_score(y_train_raw, train_preds))
                gap = train_accuracy - accuracy
                overfit = gap > 0.03
                f1 = float(f1_score(y_train_raw, train_preds, average="weighted"))
                metrics[name] = {
                    "CV_Accuracy": accuracy,
                    "CV_Accuracy_STD": score_std,
                    "Train_Accuracy": train_accuracy,
                    "Train_F1": f1,
                }
                overfit_report[name] = {
                    "train_score": train_accuracy,
                    "validation_score": accuracy,
                    "gap": gap,
                    "overfitting": overfit,
                }
                cv_scores_by_model[name] = {
                    "cv_mean": score,
                    "cv_std": score_std,
                    "display_metric": score,
                }
                oof_label_preds_by_model[name] = np.asarray(oof_preds, dtype=object)
                oof_preds_by_model[name] = np.asarray(pd.Series(oof_preds).astype("category").cat.codes, dtype=float)
                if binary_mode and np.isfinite(oof_pos_proba).any():
                    oof_pos_proba_by_model[name] = oof_pos_proba
                adjusted = self._stability_adjusted_score(score, score_std, problem_type=str(problem_type))
                model_score_for_ranking[name] = adjusted

            logger.info("Completed evaluation for %s | cv_mean=%.6f", name, score)

            adjusted_score = self._stability_adjusted_score(score, score_std, problem_type=str(problem_type))
            if adjusted_score > best_adjusted_score + 1e-12:
                best_score = score
                best_adjusted_score = adjusted_score
                best_name = name
                best_model = model

        ranked = sorted(cv_scores_by_model.items(), key=lambda item: item[1]["cv_mean"], reverse=True)
        keep_count = 6 if problem_type == "classification" else 3
        top_model_names = [name for name, _ in ranked[:keep_count]]
        filtered_models = {name: trained_models[name] for name in top_model_names if name in trained_models}
        if filtered_models:
            state["trained_models"] = filtered_models

        corr_matrix = self._build_corr_matrix(oof_preds_by_model)
        state["model_prediction_correlation"] = corr_matrix
        diversity_threshold = 0.995 if problem_type == "classification" else 0.95
        diverse_kept, dropped_correlated = self._select_diverse_models(
            model_scores=model_score_for_ranking,
            corr_matrix=corr_matrix,
            threshold=diversity_threshold,
        )
        state["dropped_correlated_models"] = dropped_correlated

        ensemble_score = float("-inf")
        ensemble_std = 0.0
        min_ensemble_improvement_pct = 0.01 if problem_type == "classification" else 0.1
        min_round_improvement_pct = 0.01
        if problem_type == "regression" and len(diverse_kept) >= 2:
            X_meta = np.column_stack([oof_preds_by_model[name] for name in diverse_kept])
            y_meta = np.asarray(y_train_raw, dtype=float)
            best_alpha = self._optimize_stacking_alpha(X_meta, y_meta, cv_splits=effective_cv_splits, objective=metric_objective)
            fold_scores = []
            oof_ens = np.zeros(len(y_meta), dtype=float)
            for tr, va in effective_cv_splits:
                meta = Ridge(alpha=best_alpha, random_state=42)
                meta.fit(X_meta[tr], y_meta[tr])
                pred = meta.predict(X_meta[va])
                oof_ens[va] = pred
                fold_scores.append(self._score_from_objective(metric_objective, y_meta[va], pred))
            ensemble_score = float(np.mean(fold_scores))
            ensemble_std = float(np.std(fold_scores))

            best_single_metric = self._convert_score_to_metric(metric_objective, best_score)
            ensemble_metric = self._convert_score_to_metric(metric_objective, ensemble_score)
            ensemble_improvement_pct = float(((best_single_metric - ensemble_metric) / max(abs(best_single_metric), 1e-12)) * 100.0)
            state["ensemble_cv_score"] = ensemble_score
            state["ensemble_cv_std"] = ensemble_std
            state["ensemble_improvement_pct"] = ensemble_improvement_pct

            if ensemble_improvement_pct < 0.5:
                state["ensemble_no_improvement_rounds"] = int(state.get("ensemble_no_improvement_rounds", 0)) + 1
            else:
                state["ensemble_no_improvement_rounds"] = 0

            if ensemble_score > best_score:
                base_models = {name: trained_models[name] for name in diverse_kept}
                meta_final = Ridge(alpha=best_alpha, random_state=42)
                meta_final.fit(X_meta, y_meta)
                best_name = "RidgeStackingEnsemble"
                best_model = RidgeStackingEnsembleRegressor(
                    base_models=base_models,
                    meta_model=meta_final,
                    target_transformed=target_transformed,
                )
                best_score = ensemble_score
                metrics[best_name] = {
                    f"CV_{self._objective_label(metric_objective)}": ensemble_metric,
                    f"CV_{self._objective_label(metric_objective)}_STD": ensemble_std,
                }
                cv_scores_by_model[best_name] = {
                    "cv_mean": ensemble_score,
                    "cv_std": ensemble_std,
                    "display_metric": ensemble_metric,
                }
                state.setdefault("reasoning_trace", []).append(
                    f"Stacking ensemble selected (alpha={best_alpha:.5f}, improvement={ensemble_improvement_pct:.4f}%)"
                )
        elif problem_type == "classification" and len(diverse_kept) >= 2:
            y_true = np.asarray(y_train_raw)
            classes_ = np.asarray(pd.Series(y_true).dropna().unique())
            n_classes = len(classes_)
            if n_classes >= 2:
                y_proba_by_model: Dict[str, np.ndarray] = {}
                for model_name in diverse_kept:
                    proba_pos = np.asarray(oof_pos_proba_by_model.get(model_name, np.array([], dtype=float)), dtype=float)
                    if proba_pos.shape[0] == len(y_true) and np.isfinite(proba_pos).any() and n_classes == 2:
                        positive_class = self._choose_positive_class(classes_)
                        neg_class = classes_[0] if classes_[0] != positive_class else classes_[1]
                        model_proba = np.zeros((len(y_true), 2), dtype=float)
                        pos_idx = 0 if classes_[0] == positive_class else 1
                        neg_idx = 1 - pos_idx
                        clipped = np.clip(proba_pos, 0.0, 1.0)
                        model_proba[:, pos_idx] = clipped
                        model_proba[:, neg_idx] = 1.0 - clipped
                        y_proba_by_model[model_name] = model_proba
                    else:
                        preds = np.asarray(oof_label_preds_by_model.get(model_name, np.array([], dtype=object)))
                        if preds.shape[0] != len(y_true):
                            continue
                        labels = pd.Series(preds).astype("category")
                        label_values = np.asarray(labels.cat.categories)
                        model_proba = np.zeros((len(y_true), n_classes), dtype=float)
                        codes = np.asarray(labels.cat.codes, dtype=int)
                        for class_idx, class_value in enumerate(classes_):
                            hits = np.where(label_values == class_value)[0]
                            if len(hits) > 0:
                                model_proba[:, class_idx] = (codes == int(hits[0])).astype(float)
                        y_proba_by_model[model_name] = model_proba

                if len(y_proba_by_model) >= 2:
                    min_score = min(float(model_score_for_ranking.get(name, 0.0)) for name in y_proba_by_model.keys())
                    offset = abs(min_score) + 1e-6 if min_score <= 0 else 0.0
                    weights = {
                        name: float(model_score_for_ranking.get(name, 0.0) + offset)
                        for name in y_proba_by_model.keys()
                    }
                    weight_sum = sum(weights.values())
                    if weight_sum <= 0.0:
                        weights = {name: 1.0 for name in y_proba_by_model.keys()}
                        weight_sum = float(len(weights))

                    model_names = list(y_proba_by_model.keys())
                    proba_stack = np.stack([y_proba_by_model[name] for name in model_names], axis=0)
                    current_weights = np.asarray([weights[name] for name in model_names], dtype=float)
                    current_weights = current_weights / max(float(current_weights.sum()), 1e-12)

                    def _ensemble_proba(weight_vector: np.ndarray) -> np.ndarray:
                        return np.tensordot(weight_vector, proba_stack, axes=(0, 0))

                    ens_proba = _ensemble_proba(current_weights)
                    ens_pred = classes_[np.argmax(ens_proba, axis=1)]
                    best_weighted_acc = float(accuracy_score(y_true, ens_pred))

                    improved_weights = False
                    for _ in range(24):
                        found_better = False
                        for idx in range(len(model_names)):
                            for delta in (0.10, -0.10):
                                trial_weights = current_weights.copy()
                                trial_weights[idx] = max(0.01, float(trial_weights[idx] + delta))
                                trial_weights = trial_weights / max(float(trial_weights.sum()), 1e-12)
                                trial_proba = _ensemble_proba(trial_weights)
                                trial_pred = classes_[np.argmax(trial_proba, axis=1)]
                                trial_acc = float(accuracy_score(y_true, trial_pred))
                                if trial_acc > best_weighted_acc + 1e-6:
                                    best_weighted_acc = trial_acc
                                    current_weights = trial_weights
                                    ens_proba = trial_proba
                                    ens_pred = trial_pred
                                    found_better = True
                                    improved_weights = True
                        if not found_better:
                            break

                    weights = {name: float(current_weights[i]) for i, name in enumerate(model_names)}
                    hc_order = sorted(model_names, key=lambda nm: float(weights.get(nm, 0.0)), reverse=True)
                    ensemble_threshold = 0.5
                    positive_class = self._choose_positive_class(classes_)
                    if n_classes == 2:
                        pos_idx = int(np.where(classes_ == positive_class)[0][0])
                        platt_a = None
                        platt_b = None
                        try:
                            y_bin = (y_true == positive_class).astype(int)
                            calib = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42)
                            calib.fit(ens_proba[:, pos_idx].reshape(-1, 1), y_bin)
                            calibrated_pos = calib.predict_proba(ens_proba[:, pos_idx].reshape(-1, 1))[:, 1]
                            calibrated_pred = np.where(calibrated_pos >= 0.5, positive_class, classes_[0] if classes_[0] != positive_class else classes_[1])
                            calibrated_acc = float(accuracy_score(y_true, calibrated_pred))
                            raw_acc = float(accuracy_score(y_true, ens_pred))
                            if calibrated_acc > raw_acc + 1e-4:
                                ens_proba[:, pos_idx] = calibrated_pos
                                ens_proba[:, 1 - pos_idx] = 1.0 - calibrated_pos
                                ens_pred = calibrated_pred
                                platt_a = float(calib.coef_[0][0])
                                platt_b = float(calib.intercept_[0])
                                state.setdefault("reasoning_trace", []).append(
                                    f"Applied Platt calibration to soft-voting probabilities (acc {raw_acc:.6f} -> {calibrated_acc:.6f})"
                                )
                        except Exception:
                            pass
                        candidate_threshold, candidate_acc = self._optimize_binary_threshold(
                            y_true=y_true,
                            pos_proba=ens_proba[:, pos_idx],
                            positive_class=positive_class,
                        )
                        base_acc = float(accuracy_score(y_true, ens_pred))
                        if np.isfinite(candidate_acc) and candidate_acc > base_acc + 1e-4:
                            ensemble_threshold = float(candidate_threshold)
                            neg_label = classes_[0] if classes_[0] != positive_class else classes_[1]
                            ens_pred = np.where(ens_proba[:, pos_idx] >= ensemble_threshold, positive_class, neg_label)
                    ensemble_score = float(accuracy_score(y_true, ens_pred))

                    fold_acc = []
                    for _, va in effective_cv_splits:
                        fold_acc.append(float(accuracy_score(y_true[va], ens_pred[va])))
                    ensemble_std = float(np.std(np.asarray(fold_acc, dtype=float)))

                    ensemble_improvement_pct = float(((ensemble_score - best_score) / max(abs(best_score), 1e-12)) * 100.0)
                    state["ensemble_cv_score"] = ensemble_score
                    state["ensemble_cv_std"] = ensemble_std
                    state["ensemble_improvement_pct"] = ensemble_improvement_pct

                    if ensemble_improvement_pct < min_ensemble_improvement_pct:
                        state["ensemble_no_improvement_rounds"] = int(state.get("ensemble_no_improvement_rounds", 0)) + 1
                    else:
                        state["ensemble_no_improvement_rounds"] = 0

                    if ensemble_score > best_score:
                        base_models = {name: trained_models[name] for name in y_proba_by_model.keys() if name in trained_models}
                        best_name = "SoftVotingEnsembleClassifier"
                        best_model = SoftVotingEnsembleClassifier(
                            base_models=base_models,
                            classes_=classes_,
                            weights=weights,
                            threshold=ensemble_threshold,
                            positive_class=positive_class if n_classes == 2 else None,
                            platt_a=platt_a,
                            platt_b=platt_b,
                        )
                        best_score = ensemble_score
                        metrics[best_name] = {
                            "CV_Accuracy": ensemble_score,
                            "CV_Accuracy_STD": ensemble_std,
                        }
                        cv_scores_by_model[best_name] = {
                            "cv_mean": ensemble_score,
                            "cv_std": ensemble_std,
                            "display_metric": ensemble_score,
                        }
                        if improved_weights:
                            state.setdefault("reasoning_trace", []).append(
                                f"Soft-voting hill climbing improved weighted ensemble accuracy to {best_weighted_acc:.6f}"
                            )
                        state.setdefault("reasoning_trace", []).append(
                            f"Soft-voting ensemble selected (improvement={ensemble_improvement_pct:.4f}%)"
                        )

                    if n_classes == 2:
                        ridge_result = self._evaluate_rank_ridge_binary(
                            y_true=y_true,
                            y_proba_by_model=y_proba_by_model,
                            model_order=hc_order,
                            classes_=classes_,
                            positive_class=positive_class,
                            cv_splits=effective_cv_splits,
                        )
                        if ridge_result is not None:
                            ridge_score = float(ridge_result["score"])
                            ridge_std = float(ridge_result["score_std"])
                            ridge_threshold = float(ridge_result["threshold"])
                            ridge_alpha = float(ridge_result["alpha"])
                            ridge_top_k = int(ridge_result["top_k"])
                            ridge_oof_pos = np.asarray(ridge_result["oof_pos"], dtype=float)

                            state["ridge_ensemble_cv_score"] = ridge_score
                            state["ridge_ensemble_cv_std"] = ridge_std
                            state["ridge_ensemble_alpha"] = ridge_alpha
                            state["ridge_ensemble_top_k"] = ridge_top_k

                            current_best_std = float(
                                (cv_scores_by_model.get(best_name, {}) or {}).get("cv_std", np.inf)
                            )
                            prefer_ridge = (
                                ridge_score > ensemble_score + 1e-4
                                or (
                                    ridge_score >= ensemble_score - 3e-4
                                    and ridge_std < ensemble_std - 2e-5
                                )
                            )
                            can_replace_current_best = (
                                ridge_score > best_score + 1e-4
                                or (
                                    ridge_score >= best_score - 3e-4
                                    and ridge_std < current_best_std - 2e-5
                                )
                            )
                            if prefer_ridge and can_replace_current_best:
                                selected_models = list(ridge_result["used_models"])
                                base_models = {name: trained_models[name] for name in selected_models if name in trained_models}
                                best_name = "RankRidgeStackingClassifier"
                                best_model = RankRidgeStackingClassifier(
                                    base_models=base_models,
                                    classes_=classes_,
                                    model_order=selected_models,
                                    ridge_model=ridge_result["ridge_model"],
                                    threshold=ridge_threshold,
                                    positive_class=positive_class,
                                )
                                best_score = ridge_score
                                metrics[best_name] = {
                                    "CV_Accuracy": ridge_score,
                                    "CV_Accuracy_STD": ridge_std,
                                }
                                cv_scores_by_model[best_name] = {
                                    "cv_mean": ridge_score,
                                    "cv_std": ridge_std,
                                    "display_metric": ridge_score,
                                }
                                oof_pos_proba_by_model[best_name] = ridge_oof_pos
                                state.setdefault("reasoning_trace", []).append(
                                    f"Rank-ridge stack selected (top_k={ridge_top_k}, alpha={ridge_alpha:.2f}, cv={ridge_score:.6f}, std={ridge_std:.6f}) due to better generalization stability"
                                )
                            else:
                                state.setdefault("reasoning_trace", []).append(
                                    f"Rank-ridge stack evaluated (top_k={ridge_top_k}, alpha={ridge_alpha:.2f}, cv={ridge_score:.6f}, std={ridge_std:.6f}) but soft-voting remained preferred"
                                )
                        else:
                            state.setdefault("reasoning_trace", []).append(
                                "Rank-ridge stack skipped: insufficient valid probabilistic base predictions"
                            )

        previous_best_score = state.get("best_score_value", float("-inf"))
        if np.isfinite(previous_best_score):
            delta = float(best_score - previous_best_score)
            relative_improvement_pct = float((delta / max(abs(previous_best_score), 1e-12)) * 100.0)
        else:
            relative_improvement_pct = 100.0

        if relative_improvement_pct < min_round_improvement_pct:
            state["no_improvement_rounds"] = int(state.get("no_improvement_rounds", 0)) + 1
        else:
            state["no_improvement_rounds"] = 0
        if (not np.isfinite(previous_best_score)) or (best_score > previous_best_score):
            state["best_score_value"] = float(best_score)

        prior_best_name = state.get("best_model_name")
        prior_best_model = state.get("best_model_object")
        prior_best_score = float(state.get("best_score_value", float("-inf")))
        if (
            prior_best_model is not None
            and isinstance(prior_best_name, str)
            and prior_best_name
            and np.isfinite(prior_best_score)
            and prior_best_score > best_score
        ):
            state.setdefault("reasoning_trace", []).append(
                f"Retained previous global best model {prior_best_name} ({prior_best_score:.6f}) over weaker current-round best ({best_score:.6f})"
            )
            best_name = prior_best_name
            best_model = prior_best_model
            best_score = prior_best_score
            if best_name not in cv_scores_by_model:
                cv_scores_by_model[best_name] = {
                    "cv_mean": float(best_score),
                    "cv_std": 0.0,
                    "display_metric": float(best_score),
                }

        state["metric_delta_pct"] = float(relative_improvement_pct)
        timeline = [*state.get("improvement_timeline", [])]
        timeline.append(
            {
                "iteration": int(state.get("iteration_count", 0)),
                "best_model": best_name,
                "best_score": float(best_score),
                "delta_pct": float(relative_improvement_pct),
            }
        )
        state["improvement_timeline"] = timeline

        stagnation_log = [*state.get("stagnation_log", [])]
        stagnation_log.append(
            f"iteration={int(state.get('iteration_count', 0))} delta_pct={relative_improvement_pct:.4f}% no_improvement_rounds={state['no_improvement_rounds']}"
        )
        state["stagnation_log"] = stagnation_log

        if int(state.get("ensemble_no_improvement_rounds", 0)) >= 3:
            state["pending_action"] = "finalize"
            state.setdefault("reasoning_trace", []).append(
                f"Stopping rule: ensemble CV improvement <{min_ensemble_improvement_pct:.2f}% for 3 rounds; finalizing to avoid leaderboard chasing"
            )

        if problem_type == "classification" and best_name in oof_pos_proba_by_model:
            y_true = np.asarray(y_train_raw)
            classes = pd.Series(y_true).dropna().unique()
            if len(classes) == 2:
                positive_class = self._choose_positive_class(np.asarray(classes))
                best_threshold, threshold_acc = self._optimize_binary_threshold(
                    y_true=y_true,
                    pos_proba=oof_pos_proba_by_model[best_name],
                    positive_class=positive_class,
                )
                baseline_acc = float(cv_scores_by_model.get(best_name, {}).get("cv_mean", best_score))
                threshold_gain = float(threshold_acc - baseline_acc) if np.isfinite(threshold_acc) else float("-inf")
                if np.isfinite(threshold_acc) and threshold_gain > 1e-4 and abs(best_threshold - 0.5) >= 0.02:
                    state["classification_threshold"] = float(best_threshold)
                    state["positive_class_label"] = positive_class
                    state.setdefault("reasoning_trace", []).append(
                        f"Optimized binary threshold for {best_name}: threshold={best_threshold:.2f}, cv_accuracy={threshold_acc:.6f}, gain={threshold_gain:.6f}"
                    )
                else:
                    state["classification_threshold"] = 0.5
                    state["positive_class_label"] = positive_class
                    state.setdefault("reasoning_trace", []).append(
                        f"Threshold optimization skipped for {best_name}: no CV gain over default 0.5 (gain={threshold_gain:.6f})"
                    )

        if problem_type == "classification":
            try:
                y_true = np.asarray(y_train_raw)
                best_preds = np.asarray(best_model.predict(X_train_raw), dtype=object) if best_model is not None else np.asarray([])
                if best_preds.shape[0] == len(y_true):
                    categorical_cols = [c for c in (state.get("categorical_columns") or []) if c in X_train_raw.columns]
                    slice_rows: list[Dict[str, Any]] = []
                    for col in categorical_cols[:3]:
                        series = pd.Series(X_train_raw[col]).astype(str).fillna("__NA__")
                        grouped = pd.DataFrame({
                            "feature_value": series,
                            "y_true": y_true,
                            "y_pred": best_preds,
                        }).groupby("feature_value")
                        for value, frame in grouped:
                            count = int(len(frame))
                            if count < 500:
                                continue
                            acc = float(accuracy_score(frame["y_true"], frame["y_pred"]))
                            slice_rows.append(
                                {
                                    "feature": str(col),
                                    "value": str(value),
                                    "count": count,
                                    "accuracy": acc,
                                }
                            )
                    if slice_rows:
                        worst = sorted(slice_rows, key=lambda x: (x["accuracy"], -x["count"]))[:8]
                        state["slice_diagnostics"] = worst
                        state.setdefault("reasoning_trace", []).append(
                            f"Computed slice diagnostics: {len(worst)} worst slices captured for specialist analysis"
                        )
            except Exception:
                pass

        if problem_type == "regression" and best_name in oof_preds_by_model:
            y_true = np.asarray(y_train_raw, dtype=float)
            y_pred = np.asarray(oof_preds_by_model[best_name], dtype=float)
            residuals = y_true - y_pred
            residual_mean = float(np.mean(residuals))
            residual_var = float(np.var(residuals))
            try:
                hetero_corr = float(np.corrcoef(np.abs(residuals), np.abs(y_pred))[0, 1])
            except Exception:
                hetero_corr = 0.0
            if np.isnan(hetero_corr):
                hetero_corr = 0.0
            pattern = abs(hetero_corr) >= 0.2
            state["residual_mean"] = residual_mean
            state["residual_variance"] = residual_var
            state["residual_analysis"] = {
                "mean_residual": residual_mean,
                "residual_variance": residual_var,
                "heteroscedasticity_corr": hetero_corr,
            }
            state["residual_pattern_detected"] = pattern
            state["residual_action_recommended"] = pattern

        state["current_metrics"] = metrics
        state["overfitting_report"] = overfit_report
        state["best_model_name"] = best_name
        state["best_model_object"] = best_model
        state["metric_evaluation_split"] = "cross_validation"
        state["cv_scores"] = cv_scores_by_model
        state["top_cv_models"] = top_model_names

        if problem_type == "regression":
            metric_label = self._objective_label(metric_objective)
            state.setdefault("reasoning_trace", []).append(
                f"Best model by {len(effective_cv_splits)}-fold CV: {best_name} ({metric_label}={self._convert_score_to_metric(metric_objective, best_score):.6f}, delta={relative_improvement_pct:.4f}%)"
            )
        else:
            state.setdefault("reasoning_trace", []).append(
                f"Best model by {len(effective_cv_splits)}-fold CV: {best_name} (Accuracy={best_score:.4f}, delta={relative_improvement_pct:.4f}%)"
            )

        return state
