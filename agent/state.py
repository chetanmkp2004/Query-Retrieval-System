"""State definitions for LangGraph workflow."""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentGraphState(TypedDict, total=False):
    file_path: str
    test_file_path: str
    sample_submission_file_path: str
    submission_file_name: str
    submission_path: str
    target_column: str
    dataset_summary: Dict[str, Any]
    problem_type: str
    preprocessing_done: bool
    models_tried: List[str]
    current_metrics: Dict[str, Dict[str, float]]
    metric_evaluation_split: str
    best_model_name: str
    best_model_object: Any
    tuning_performed: bool
    iteration_count: int
    max_iterations: int
    reasoning_trace: List[str]
    workflow_trace: List[str]
    llm_decision_audit: List[Dict[str, Any]]
    decision_source: str
    show_langgraph: bool
    suggested_models: List[str]
    suggested_hyperparams: Dict[str, Dict[str, List[Any]]]
    strategy_reason: str
    preprocess_strategy: Dict[str, Any]
    preprocess_strategy_reason: str
    metric_objective: str
    encoding_choice: str
    encoding_cv_scores: Dict[str, float]
    mestimate_smoothing: float
    overfitting_report: Dict[str, Dict[str, float | bool]]
    engineered_feature_columns: List[str]
    engineered_feature_count: int
    engineered_base_columns: List[str]
    engineered_aggregate_base_columns: List[str]
    engineered_feature_metadata: Dict[str, Any]
    feature_pruning_decision: str
    feature_expansion_decisions: Dict[str, Any]
    features_added_this_round: bool
    residual_pattern_detected: bool
    residual_analysis: Dict[str, Any]
    residual_variance: float
    residual_mean: float
    residual_action_recommended: bool
    apply_feature_engineering: bool
    feature_engineering_reason: str
    competition_context: str
    competition_context_source: str

    raw_df: Any
    X_train: Any
    X_val: Any
    X_test: Any
    X_train_raw: Any
    X_holdout_raw: Any
    y_train: Any
    y_val: Any
    y_test: Any
    y_train_raw: Any
    y_holdout: Any
    preprocessor: Any
    transformed_feature_names: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    train_row_indices: List[int]
    holdout_row_indices: List[int]
    cv_split_indices: List[Any]
    trained_models: Dict[str, Any]
    model_params: Dict[str, Dict[str, Any]]
    training_time_estimates: Dict[str, float]
    pending_action: str
    no_improvement_rounds: int
    best_score_value: float
    metric_delta_pct: float
    improvement_timeline: List[Dict[str, Any]]
    stagnation_log: List[str]
    ensemble_cv_score: float
    ensemble_cv_std: float
    ensemble_improvement_pct: float
    ensemble_no_improvement_rounds: int
    model_prediction_correlation: Dict[str, Dict[str, float]]
    dropped_correlated_models: List[str]
    final_feature_importance: List[Dict[str, float]]
    force_tune: bool
    target_transform_decision: str
    target_transform_applied: bool
    target_skewness: float
    target_transform_comparison: Dict[str, float]
    tuned_model_types: List[str]
    tuning_pass_count: int
    max_tuning_passes: int
    final_holdout_metrics: Dict[str, float]
    holdout_evaluated_once: bool
    model_artifact_path: str
    report_artifact_path: str
    classification_threshold: float
    positive_class_label: Any
    run_timing: Dict[str, float]
    slice_diagnostics: List[Dict[str, Any]]
    enable_pseudo_labeling: bool
    pseudo_label_min_confidence: float
    pseudo_label_min_count: int


def initialize_state(file_path: str, target_column: str, max_iterations: int) -> AgentGraphState:
    return AgentGraphState(
        file_path=file_path,
        test_file_path="",
        sample_submission_file_path="",
        submission_file_name="submission.csv",
        submission_path="",
        target_column=target_column,
        dataset_summary={},
        problem_type="",
        preprocessing_done=False,
        models_tried=[],
        current_metrics={},
        best_model_name="",
        best_model_object=None,
        tuning_performed=False,
        iteration_count=0,
        max_iterations=max_iterations,
        reasoning_trace=[],
        workflow_trace=[],
        llm_decision_audit=[],
        decision_source="",
        show_langgraph=False,
        suggested_models=[],
        suggested_hyperparams={},
        strategy_reason="",
        preprocess_strategy={},
        preprocess_strategy_reason="",
        metric_objective="",
        encoding_choice="onehot",
        encoding_cv_scores={},
        mestimate_smoothing=10.0,
        overfitting_report={},
        engineered_feature_columns=[],
        engineered_feature_count=0,
        engineered_base_columns=[],
        engineered_aggregate_base_columns=[],
        engineered_feature_metadata={},
        feature_pruning_decision="",
        feature_expansion_decisions={},
        features_added_this_round=False,
        residual_pattern_detected=False,
        residual_analysis={},
        residual_variance=0.0,
        residual_mean=0.0,
        residual_action_recommended=False,
        apply_feature_engineering=False,
        feature_engineering_reason="",
        competition_context="",
        competition_context_source="",
        X_train_raw=None,
        X_holdout_raw=None,
        y_train_raw=None,
        y_holdout=None,
        train_row_indices=[],
        holdout_row_indices=[],
        cv_split_indices=[],
        trained_models={},
        model_params={},
        training_time_estimates={},
        pending_action="train",
        no_improvement_rounds=0,
        best_score_value=float("-inf"),
        metric_delta_pct=0.0,
        improvement_timeline=[],
        stagnation_log=[],
        ensemble_cv_score=float("-inf"),
        ensemble_cv_std=0.0,
        ensemble_improvement_pct=0.0,
        ensemble_no_improvement_rounds=0,
        model_prediction_correlation={},
        dropped_correlated_models=[],
        final_feature_importance=[],
        force_tune=False,
        target_transform_decision="none",
        target_transform_applied=False,
        target_skewness=0.0,
        target_transform_comparison={},
        tuned_model_types=[],
        tuning_pass_count=0,
        max_tuning_passes=3,
        final_holdout_metrics={},
        holdout_evaluated_once=False,
        model_artifact_path="",
        report_artifact_path="",
        classification_threshold=0.5,
        positive_class_label=None,
        run_timing={},
        slice_diagnostics=[],
        enable_pseudo_labeling=True,
        pseudo_label_min_confidence=0.995,
        pseudo_label_min_count=2000,
    )
