"""LangGraph workflow wiring for the LLM ML agent."""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from agent.brain import (
    llm_decision_node,
    llm_feature_engineering_decision_node,
    llm_model_strategy_node,
    llm_preprocess_strategy_node,
)
from agent.state import AgentGraphState
from tools.evaluate import EvaluateModelTool
from tools.feature_engineering import FeatureEngineeringTool
from tools.feature_importance import FeatureImportanceTool
from tools.inspect import InspectDatasetTool
from tools.preprocess import PreprocessTool
from tools.train import DetectProblemTypeTool, TrainModelTool
from tools.tune import HyperparameterTuningTool


GRAPH_NODES = [
    "inspect_dataset",
    "llm_feature_engineering_decision",
    "feature_engineering",
    "detect_problem_type",
    "llm_model_strategy",
    "llm_preprocess_strategy",
    "preprocess",
    "train_model",
    "llm_decision_node",
    "tune_model",
    "train_alternative",
    "finalize",
]

GRAPH_EDGES = [
    ("START", "inspect_dataset"),
    ("inspect_dataset", "llm_feature_engineering_decision"),
    ("llm_feature_engineering_decision", "feature_engineering"),
    ("feature_engineering", "detect_problem_type"),
    ("llm_feature_engineering_decision", "detect_problem_type"),
    ("detect_problem_type", "llm_model_strategy"),
    ("llm_model_strategy", "llm_preprocess_strategy"),
    ("llm_preprocess_strategy", "preprocess"),
    ("preprocess", "train_model"),
    ("train_model", "llm_decision_node"),
    ("tune_model", "llm_decision_node"),
    ("train_alternative", "llm_decision_node"),
    ("finalize", "END"),
]

GRAPH_CONDITIONAL_EDGES = {
    "llm_decision_node": {
        "tune_model": "tune_model",
        "train_alternative": "train_alternative",
        "train_model": "train_model",
        "finalize": "finalize",
    }
}


def get_graph_spec() -> Dict[str, Any]:
    return {
        "nodes": GRAPH_NODES,
        "edges": GRAPH_EDGES,
        "conditional_edges": GRAPH_CONDITIONAL_EDGES,
        "safeguards": [
            "max_iterations route guard forces finalize",
            "tune path executes only when tuning_performed is False",
        ],
    }


def get_graph_mermaid() -> str:
    lines = [
        "flowchart TD",
        "  START([START]) --> inspect_dataset",
        "  inspect_dataset --> llm_feature_engineering_decision",
        "  llm_feature_engineering_decision -- yes --> feature_engineering",
        "  llm_feature_engineering_decision -- no --> detect_problem_type",
        "  feature_engineering --> detect_problem_type",
        "  detect_problem_type --> llm_model_strategy",
        "  llm_model_strategy --> llm_preprocess_strategy",
        "  llm_preprocess_strategy --> preprocess",
        "  preprocess --> train_model",
        "  train_model --> llm_decision_node",
        "  llm_decision_node -- train --> train_model",
        "  llm_decision_node -- tune --> tune_model",
        "  llm_decision_node -- try_alternative_model --> train_alternative",
        "  llm_decision_node -- finalize --> finalize",
        "  tune_model --> llm_decision_node",
        "  train_alternative --> llm_decision_node",
        "  finalize --> END([END])",
    ]
    return "\n".join(lines)


def build_graph(llm: Any):
    inspect_tool = InspectDatasetTool()
    feature_engineering_tool = FeatureEngineeringTool()
    detect_tool = DetectProblemTypeTool()
    preprocess_tool = PreprocessTool()
    train_tool = TrainModelTool()
    evaluate_tool = EvaluateModelTool()
    tune_tool = HyperparameterTuningTool()
    fi_tool = FeatureImportanceTool()

    def inspect_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = inspect_tool.invoke({"state": state})
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "inspect_dataset"]
        return updated

    def llm_fe_decision_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = llm_feature_engineering_decision_node(state, llm=llm)
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "llm_feature_engineering_decision"]
        return updated

    def detect_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = detect_tool.invoke({"state": state})
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "detect_problem_type"]
        return updated

    def feature_engineering_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = feature_engineering_tool.invoke({"state": state})
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "feature_engineering"]
        return updated

    def preprocess_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = preprocess_tool.invoke({"state": state})
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "preprocess"]
        return updated

    def llm_strategy_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = llm_model_strategy_node(state, llm=llm)
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "llm_model_strategy"]
        return updated

    def llm_preprocess_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = llm_preprocess_strategy_node(state, llm=llm)
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "llm_preprocess_strategy"]
        return updated

    def train_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = train_tool.invoke({"state": {**state, "pending_action": "train"}})
        updated = evaluate_tool.invoke({"state": updated})
        updated["iteration_count"] = int(state.get("iteration_count", 0)) + 1
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "train_model"]
        return updated

    def llm_decide_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = llm_decision_node(state, llm=llm)
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "llm_decision_node"]
        return updated

    def tune_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = tune_tool.invoke({"state": state})
        updated = evaluate_tool.invoke({"state": updated})
        updated["iteration_count"] = int(state.get("iteration_count", 0)) + 1
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "tune_model"]
        return updated

    def train_alternative_node(state: AgentGraphState) -> Dict[str, Any]:
        updated_state = {**state, "pending_action": "try_alternative_model"}
        updated = train_tool.invoke({"state": updated_state})
        updated = evaluate_tool.invoke({"state": updated})
        updated["iteration_count"] = int(state.get("iteration_count", 0)) + 1
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "train_alternative"]
        return updated

    def finalize_node(state: AgentGraphState) -> Dict[str, Any]:
        updated = fi_tool.invoke({"state": state})
        updated["pending_action"] = "finalize"
        updated["workflow_trace"] = [*state.get("workflow_trace", []), "finalize"]
        return updated

    def route_after_decision(state: AgentGraphState) -> str:
        decision = state.get("pending_action", "finalize")
        if int(state.get("iteration_count", 0)) >= int(state.get("max_iterations", 5)):
            return "finalize"
        if int(state.get("no_improvement_rounds", 0)) >= 3:
            return "finalize"
        tuning_pass_count = int(state.get("tuning_pass_count", 0))
        max_tuning_passes = int(state.get("max_tuning_passes", 3) or 3)
        if decision == "tune" and tuning_pass_count < max_tuning_passes:
            return "tune_model"
        if decision == "try_alternative_model":
            return "train_alternative"
        if decision == "train":
            return "train_model"
        return "finalize"

    graph = StateGraph(AgentGraphState)
    graph.add_node("inspect_dataset", inspect_node)
    graph.add_node("llm_feature_engineering_decision", llm_fe_decision_node)
    graph.add_node("feature_engineering", feature_engineering_node)
    graph.add_node("detect_problem_type", detect_node)
    graph.add_node("llm_model_strategy", llm_strategy_node)
    graph.add_node("llm_preprocess_strategy", llm_preprocess_node)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("train_model", train_node)
    graph.add_node("llm_decision_node", llm_decide_node)
    graph.add_node("tune_model", tune_node)
    graph.add_node("train_alternative", train_alternative_node)
    graph.add_node("finalize", finalize_node)

    def route_feature_engineering(state: AgentGraphState) -> str:
        """Route to feature engineering if LLM recommends it, otherwise skip to problem type detection."""
        apply_fe = bool(state.get("apply_feature_engineering", False))
        return "feature_engineering" if apply_fe else "detect_problem_type"

    graph.add_edge(START, "inspect_dataset")
    graph.add_edge("inspect_dataset", "llm_feature_engineering_decision")
    graph.add_conditional_edges(
        "llm_feature_engineering_decision",
        route_feature_engineering,
        {
            "feature_engineering": "feature_engineering",
            "detect_problem_type": "detect_problem_type",
        },
    )
    graph.add_edge("feature_engineering", "detect_problem_type")
    graph.add_edge("detect_problem_type", "llm_model_strategy")
    graph.add_edge("llm_model_strategy", "llm_preprocess_strategy")
    graph.add_edge("llm_preprocess_strategy", "preprocess")
    graph.add_edge("preprocess", "train_model")
    graph.add_edge("train_model", "llm_decision_node")

    graph.add_conditional_edges(
        "llm_decision_node",
        route_after_decision,
        {
            "tune_model": "tune_model",
            "train_alternative": "train_alternative",
            "train_model": "train_model",
            "finalize": "finalize",
        },
    )

    graph.add_edge("tune_model", "llm_decision_node")
    graph.add_edge("train_alternative", "llm_decision_node")
    graph.add_edge("finalize", END)

    return graph.compile()
