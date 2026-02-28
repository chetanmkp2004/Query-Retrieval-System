"""Model training and problem type detection tools."""
from __future__ import annotations

import time
import io
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from inspect import signature
from typing import Any, ClassVar, Dict, Type
import logging

import numpy as np
import pandas as pd
from scipy import sparse
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import all_estimators
import warnings

try:
    from category_encoders import MEstimateEncoder
except Exception:  # pragma: no cover - optional dependency fallback
    MEstimateEncoder = None


logger = logging.getLogger(__name__)


class StateInput(BaseModel):
    state: Dict[str, Any] = Field(..., description="Current mutable agent state")


class DetectProblemTypeTool(BaseTool):
    name: str = "detect_problem_type"
    description: str = "Detect whether target is regression or classification"
    args_schema: ClassVar[Type[BaseModel]] = StateInput

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = state.get("raw_df")
        target_column = state.get("target_column")
        if df is None:
            raise ValueError("raw_df missing in state")

        target = df[target_column]
        if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 15:
            state["problem_type"] = "regression"
        else:
            state["problem_type"] = "classification"

        state.setdefault("reasoning_trace", []).append(
            f"Detected problem type: {state['problem_type']}"
        )
        return state


class TrainModelTool(BaseTool):
    name: str = "train_model"
    description: str = "Train baseline models based on problem type"
    args_schema: ClassVar[Type[BaseModel]] = StateInput

    class _SafeMEstimateEncoder:
        def __init__(self, cols: list[str], m: float = 10.0):
            self.cols = cols
            self.m = float(m)
            self.encoder = None

        def fit(self, X, y=None):
            if MEstimateEncoder is None:
                self.encoder = None
                return self
            self.encoder = MEstimateEncoder(cols=self.cols, m=self.m, randomized=False, handle_unknown="value", handle_missing="value")
            self.encoder.fit(pd.DataFrame(X).copy(), y)
            return self

        def transform(self, X):
            frame = pd.DataFrame(X).copy()
            if self.encoder is None:
                for col in self.cols:
                    if col in frame.columns:
                        frame[col] = frame[col].astype("category").cat.codes.astype(float)
                return frame
            return self.encoder.transform(frame)

    class _InverseFeatures:
        def __init__(self, min_abs: float = 1e-6):
            self.min_abs = min_abs

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
            safe = np.where(np.abs(arr) < self.min_abs, np.sign(arr) * self.min_abs + (arr == 0) * self.min_abs, arr)
            inv = 1.0 / safe
            if sparse.issparse(X):
                return sparse.hstack([X, sparse.csr_matrix(inv)], format="csr")
            return np.hstack([arr, inv])

    class _AppendRandomTreesEmbedding:
        def __init__(self, n_estimators: int = 32, max_depth: int = 5, random_state: int = 42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.random_state = random_state
            self.rte = RandomTreesEmbedding(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                sparse_output=True,
                random_state=self.random_state,
                n_jobs=-1,
            )

        def fit(self, X, y=None):
            self.rte.fit(X, y)
            return self

        def transform(self, X):
            emb = self.rte.transform(X)
            if sparse.issparse(X):
                return sparse.hstack([X, emb], format="csr")
            return sparse.hstack([sparse.csr_matrix(X), emb], format="csr")

    class _LabelEncodingClassifierWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, estimator: Any):
            self.estimator = estimator

        def fit(self, X, y):
            labels = np.asarray(y)
            unique = pd.Series(labels).dropna().unique().tolist()
            self.classes_ = np.asarray(unique, dtype=object)
            self._forward_map = {label: idx for idx, label in enumerate(self.classes_)}
            self._inverse_map = {idx: label for label, idx in self._forward_map.items()}
            y_encoded = np.asarray([self._forward_map[val] for val in labels], dtype=int)
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y_encoded)
            return self

        def predict(self, X):
            encoded_preds = np.asarray(self.estimator_.predict(X), dtype=int)
            return np.asarray([self._inverse_map.get(int(v), self.classes_[0]) for v in encoded_preds], dtype=object)

        def predict_proba(self, X):
            if hasattr(self.estimator_, "predict_proba"):
                return np.asarray(self.estimator_.predict_proba(X), dtype=float)
            preds = self.predict(X)
            proba = np.zeros((len(preds), len(self.classes_)), dtype=float)
            for idx, label in enumerate(self.classes_):
                proba[:, idx] = (preds == label).astype(float)
            return proba

    @staticmethod
    def _slice_rows(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return data[indices]

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
    def _build_ordered_candidates(problem_type: str, llm_suggested: list[str], pending_action: str, best_model_base: str) -> list[str]:
        ordered: list[str] = []
        for name in llm_suggested:
            clean = str(name).strip()
            if clean and clean.lower() not in {n.lower() for n in ordered}:
                ordered.append(clean)

        if pending_action == "try_alternative_model" and best_model_base:
            ordered = [name for name in ordered if name.strip().lower() != best_model_base]

        return ordered

    @staticmethod
    def _prioritize_large_scale_candidates(ordered_models: list[str], problem_type: str, n_rows: int) -> list[str]:
        if problem_type != "classification" or n_rows < 250000:
            return ordered_models

        augmented = list(ordered_models)
        existing = {str(name).strip().lower() for name in augmented}
        for candidate in ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "GaussianNB"]:
            if candidate.lower() not in existing:
                augmented.append(candidate)
                existing.add(candidate.lower())

        preferred = [
            "histgradientboostingclassifier",
            "xgbclassifier",
            "lgbmclassifier",
            "catboostclassifier",
            "logisticregression",
            "sgdclassifier",
            "gaussiannb",
        ]
        rank = {name: idx for idx, name in enumerate(preferred)}

        return sorted(
            augmented,
            key=lambda name: (rank.get(str(name).strip().lower(), 999), augmented.index(name)),
        )

    @staticmethod
    def _model_family(model_name: str, problem_type: str) -> str:
        key = model_name.lower()
        if key in {"linearregression", "logisticregression"}:
            return "linear"
        if key.startswith("ridge"):
            return "ridge"
        if key.startswith("mlp"):
            return "neural"
        tree_tokens = ("randomforest", "gradientboost", "histgradient", "xgb", "lgbm", "catboost", "decisiontree", "extratrees")
        if any(token in key for token in tree_tokens):
            return "tree"
        return "generic"

    def _onehot_preprocessor(self, numeric_cols: list[str], categorical_cols: list[str], family: str):
        num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if family in {"linear", "ridge", "neural"}:
            num_steps.append(("scaler", StandardScaler()))
        if family == "linear":
            num_steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
        if family == "ridge":
            num_steps.append(("nystroem", Nystroem(kernel="poly", degree=2, n_components=128, random_state=42)))

        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", cat_encoder),
        ]
        return ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=num_steps), numeric_cols),
                ("cat", Pipeline(steps=cat_steps), categorical_cols),
            ],
            remainder="drop",
        )

    def _build_model_pipeline(
        self,
        model_name: str,
        estimator: Any,
        numeric_cols: list[str],
        categorical_cols: list[str],
        encoding_choice: str,
        mestimate_smoothing: float,
        use_inverse_features: bool,
        use_rf_embedding: bool,
        linear_expansion_mode: str = "full",
    ):
        family = self._model_family(model_name, "")
        steps: list[tuple[str, Any]] = []

        if encoding_choice == "mestimate" and categorical_cols:
            steps.append(("mestimate", self._SafeMEstimateEncoder(cols=categorical_cols, m=mestimate_smoothing)))
            num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
            if family in {"linear", "ridge", "neural"}:
                num_steps.append(("scaler", StandardScaler()))
            if family == "linear" and linear_expansion_mode == "full":
                num_steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
            if family == "ridge":
                num_steps.append(("nystroem", Nystroem(kernel="poly", degree=2, n_components=128, random_state=42)))
            steps.append(("numeric_post", Pipeline(steps=num_steps)))
        else:
            effective_family = family
            if family == "linear" and linear_expansion_mode != "full":
                effective_family = "generic"
            steps.append(("preprocessor", self._onehot_preprocessor(numeric_cols, categorical_cols, family=effective_family)))

        if family in {"linear", "neural"} and use_inverse_features:
            steps.append(("inverse_features", self._InverseFeatures()))

        if family == "linear" and use_rf_embedding:
            steps.append(("rf_embedding", self._AppendRandomTreesEmbedding(n_estimators=24, max_depth=5, random_state=42)))

        steps.append(("model", estimator))
        return Pipeline(steps=steps)

    def _evaluate_encoding_choice(
        self,
        X_train_raw: pd.DataFrame,
        y_train: pd.Series,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        numeric_cols: list[str],
        categorical_cols: list[str],
        problem_type: str,
        m_smoothing: float,
    ) -> tuple[str, Dict[str, float]]:
        if not categorical_cols:
            return "onehot", {"onehot": 0.0}

        if problem_type == "regression":
            baseline = Ridge(random_state=42)
            scoring = "neg_mean_squared_error"
        else:
            baseline = LogisticRegression(max_iter=120, solver="saga", tol=1e-3, random_state=42)
            scoring = "accuracy"

        onehot_pipe = Pipeline(
            steps=[
                ("preprocessor", self._onehot_preprocessor(numeric_cols, categorical_cols, family="generic")),
                ("model", baseline),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            onehot_score = float(np.mean(cross_val_score(onehot_pipe, X_train_raw, y_train, cv=cv_splits, scoring=scoring, n_jobs=1)))

        scores = {"onehot": onehot_score}
        if MEstimateEncoder is not None:
            mest_pipe = Pipeline(
                steps=[
                    ("mestimate", self._SafeMEstimateEncoder(cols=categorical_cols, m=m_smoothing)),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", baseline),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                mest_score = float(np.mean(cross_val_score(mest_pipe, X_train_raw, y_train, cv=cv_splits, scoring=scoring, n_jobs=1)))
            scores["mestimate"] = mest_score

        choice = max(scores.items(), key=lambda item: item[1])[0]
        return choice, scores

    @staticmethod
    def _instantiate_estimator(estimator_cls, pending_action: str, problem_type: str):
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
            kwargs["n_estimators"] = 300
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
        if "max_depth" in init_params and "randomforest" in estimator_name:
            kwargs["max_depth"] = 20
        if "min_samples_leaf" in init_params and "randomforest" in estimator_name:
            kwargs["min_samples_leaf"] = 2
        if problem_type == "classification" and pending_action == "try_alternative_model" and "class_weight" in init_params:
            kwargs["class_weight"] = "balanced"

        if estimator_name == "logisticregression":
            if "solver" in init_params:
                kwargs["solver"] = "saga"
            if "max_iter" in init_params:
                kwargs["max_iter"] = 200
            if "tol" in init_params:
                kwargs["tol"] = 1e-3

        if estimator_name in {"sgdclassifier", "sgdregressor"}:
            if "early_stopping" in init_params:
                kwargs["early_stopping"] = True
            if "max_iter" in init_params:
                kwargs["max_iter"] = 200
            if "tol" in init_params:
                kwargs["tol"] = 1e-3

        estimator_name = getattr(estimator_cls, "__name__", "").lower()
        if problem_type == "regression" and estimator_name == "mlpregressor":
            return MLPRegressor(
                hidden_layer_sizes=(256, 64, 16, 4),
                activation="relu",
                early_stopping=True,
                alpha=0.0005,
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
            )

        estimator = estimator_cls(**kwargs)
        if problem_type == "classification" and estimator_name == "xgbclassifier":
            estimator = TrainModelTool._LabelEncodingClassifierWrapper(estimator)
        return estimator

    def _estimate_training_seconds(self, model, X_train, y_train, random_state: int) -> float:
        n_rows = int(getattr(X_train, "shape", [len(y_train), 1])[0])
        if n_rows <= 0:
            return 0.0

        sample_rows = min(max(500, int(n_rows * 0.1)), 5000)
        if sample_rows >= n_rows:
            sample_rows = max(1, n_rows // 2)

        rng = np.random.default_rng(random_state)
        sampled_indices = np.sort(rng.choice(n_rows, size=sample_rows, replace=False))

        X_sample = self._slice_rows(X_train, sampled_indices)
        y_sample = self._slice_rows(y_train, sampled_indices)

        probe_model = clone(model)
        start = time.perf_counter()
        probe_model.fit(X_sample, y_sample)
        sample_time = time.perf_counter() - start

        scaled_estimate = sample_time * (n_rows / sample_rows)
        return round(float(scaled_estimate), 3)

    @staticmethod
    def _should_skip_training_estimate(model_name: str, n_rows: int) -> bool:
        heavy_model_tokens = ("xgb", "lgbm", "catboost")
        lowered = model_name.lower()
        if any(token in lowered for token in heavy_model_tokens):
            return True
        if n_rows > 60000:
            return True
        return False

    @staticmethod
    def _skip_model_for_budget(model_name: str, n_rows: int, n_cols: int) -> str | None:
        lowered = model_name.lower()
        if n_rows < 250000:
            return None

        if "histgradientboosting" in lowered:
            return None

        very_slow_tokens = (
            "gradientboostingclassifier",
            "gradientboostingregressor",
            "randomforest",
            "extratrees",
            "kneighbors",
            "gaussianprocess",
            "svc",
            "svr",
        )
        if any(token in lowered for token in very_slow_tokens):
            return f"skipped by compute budget for very large dataset ({n_rows} rows)"

        if n_rows >= 400000 and n_cols >= 100 and "logisticregression" in lowered:
            return f"skipped by compute budget for high-dimensional very large dataset ({n_rows}x{n_cols})"

        return None

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        X_train = state.get("X_train_raw")
        y_train = state.get("y_train")
        y_train_raw = state.get("y_train_raw")
        problem_type = state.get("problem_type")
        if X_train is None or y_train is None:
            raise ValueError("Training matrices are missing")

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train_raw must be a DataFrame for model-specific preprocessing")

        n_rows = int(getattr(X_train, "shape", [len(y_train), 1])[0])

        numeric_cols = [c for c in state.get("numeric_columns", []) if c in X_train.columns]
        categorical_cols = [c for c in state.get("categorical_columns", []) if c in X_train.columns]
        cv_splits = [
            (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
            for tr, va in (state.get("cv_split_indices") or [])
        ]
        if not cv_splits:
            raise ValueError("cv_split_indices missing for fold-safe training")

        m_smoothing = float(state.get("mestimate_smoothing", 10.0) or 10.0)
        encoding_choice, encoding_scores = self._evaluate_encoding_choice(
            X_train_raw=X_train,
            y_train=y_train_raw if y_train_raw is not None else y_train,
            cv_splits=cv_splits,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            problem_type=problem_type,
            m_smoothing=m_smoothing,
        )
        state["encoding_choice"] = encoding_choice
        state["encoding_cv_scores"] = encoding_scores
        state.setdefault("reasoning_trace", []).append(
            f"Encoding choice via CV: {encoding_choice} | scores={encoding_scores}"
        )

        pending_action = state.get("pending_action", "train")
        suggested_models = [str(name) for name in state.get("suggested_models", [])]
        candidates: Dict[str, Any] = {}
        skipped_models: list[str] = []
        n_cols = int(X_train.shape[1])
        if str(problem_type) == "classification" and n_rows >= 250000:
            max_models_to_train = 6
        elif str(problem_type) == "classification":
            max_models_to_train = 5
        else:
            max_models_to_train = 4

        if not suggested_models:
            raise ValueError("LLM did not provide suggested_models; cannot proceed with strictly LLM-driven model selection")

        best_model_name = str(state.get("best_model_name") or "")
        best_model_base = best_model_name.replace(" (tuned)", "").strip().lower()
        ordered_models = self._build_ordered_candidates(
            problem_type=problem_type,
            llm_suggested=suggested_models,
            pending_action=pending_action,
            best_model_base=best_model_base,
        )
        ordered_models = self._prioritize_large_scale_candidates(
            ordered_models=ordered_models,
            problem_type=str(problem_type),
            n_rows=n_rows,
        )

        for model_name in ordered_models:
            budget_skip_reason = self._skip_model_for_budget(model_name=model_name, n_rows=n_rows, n_cols=n_cols)
            if budget_skip_reason:
                skipped_models.append(f"{model_name} ({budget_skip_reason})")
                continue

            estimator_cls = self._resolve_estimator_class(model_name=model_name, problem_type=problem_type)
            if estimator_cls is None:
                skipped_models.append(f"{model_name} (not found/installed for {problem_type})")
                continue
            try:
                base_estimator = self._instantiate_estimator(
                    estimator_cls=estimator_cls,
                    pending_action=pending_action,
                    problem_type=problem_type,
                )
                family = self._model_family(model_name, problem_type)
                is_large_tabular = n_rows > 20000 or n_cols > 80
                is_classification = problem_type == "classification"

                linear_expansion_mode = "full"
                if family == "linear" and (is_classification or is_large_tabular):
                    linear_expansion_mode = "light"

                use_inverse = family in {"linear", "neural"}
                if family == "linear" and linear_expansion_mode != "full":
                    use_inverse = False
                if family == "neural" and is_large_tabular:
                    use_inverse = False

                use_rf_embedding = family == "linear" and linear_expansion_mode == "full"
                model_pipeline = self._build_model_pipeline(
                    model_name=model_name,
                    estimator=base_estimator,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    encoding_choice=encoding_choice,
                    mestimate_smoothing=m_smoothing,
                    use_inverse_features=use_inverse,
                    use_rf_embedding=use_rf_embedding,
                    linear_expansion_mode=linear_expansion_mode,
                )
                candidates[model_name] = model_pipeline
                if len(candidates) >= max_models_to_train:
                    break
            except Exception as exc:
                skipped_models.append(f"{model_name} ({exc})")

        if skipped_models:
            state.setdefault("reasoning_trace", []).append(
                f"Skipped unsupported/unavailable LLM models: {', '.join(skipped_models)}"
            )

        if not candidates:
            raise ValueError("No trainable LLM-suggested models were resolved from available estimators")

        trained_models: Dict[str, Any] = {}
        model_params: Dict[str, Dict[str, Any]] = {}
        training_time_estimates: Dict[str, float] = {}
        model_names = []
        random_state = int(state.get("random_state", 42))

        for name, model in candidates.items():
            try:
                n_rows = int(getattr(X_train, "shape", [len(y_train), 1])[0])
                if self._should_skip_training_estimate(name, n_rows=n_rows):
                    estimated_seconds = -1.0
                    logger.info("Training %s (estimate skipped)", name)
                    state.setdefault("reasoning_trace", []).append(
                        f"{name}: training estimate skipped for faster execution"
                    )
                else:
                    estimated_seconds = self._estimate_training_seconds(model, X_train, y_train, random_state)
                    logger.info("Training %s (estimated %.2fs)", name, estimated_seconds)

                start = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    if "lgbm" in str(name).lower():
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train)
                elapsed = time.perf_counter() - start
                logger.info("Completed training %s in %.2fs", name, elapsed)
                trained_models[name] = model
                model_names.append(name)
                training_time_estimates[name] = estimated_seconds
                params = model.get_params()
                params["estimated_train_time_seconds"] = estimated_seconds
                params["train_time_seconds"] = round(elapsed, 3)
                params["encoding_choice"] = encoding_choice
                model_params[name] = params
                state.setdefault("feature_expansion_decisions", {})[name] = {
                    "model_family": self._model_family(name, problem_type),
                    "inverse_features": self._model_family(name, problem_type) in {"linear", "neural"},
                    "random_trees_embedding": self._model_family(name, problem_type) == "linear",
                }
                if estimated_seconds >= 0:
                    state.setdefault("reasoning_trace", []).append(
                        f"{name}: estimated training {estimated_seconds:.2f}s, actual {elapsed:.2f}s"
                    )
                else:
                    state.setdefault("reasoning_trace", []).append(
                        f"{name}: actual training {elapsed:.2f}s"
                    )
            except Exception as exc:
                state.setdefault("reasoning_trace", []).append(
                    f"Skipped model {name} due to training error: {exc}"
                )

        if not trained_models:
            raise ValueError("All LLM-suggested models failed during training")

        state["trained_models"] = trained_models
        state["models_tried"] = model_names
        state["model_params"] = model_params
        state["training_time_estimates"] = training_time_estimates
        state.setdefault("reasoning_trace", []).append(
            f"Trained top LLM-selected models: {', '.join(model_names)}"
        )
        return state
