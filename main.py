"""CLI entrypoint for the LLM-powered agentic ML system."""
from __future__ import annotations

import argparse
import csv
import getpass
import os
import sys
import traceback

from dotenv import load_dotenv

from agent.runner import print_final_report, run_agent
from config.settings import DEFAULT_MODEL, configure_logging, set_openrouter_api_key, build_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM Agentic ML workflow")
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--test-file", help="Path to external test CSV for submission generation")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv", help="Path to sample submission CSV")
    parser.add_argument(
        "--competition-description",
        default="data/data_description.txt",
        help="Path to temporary competition dataset description text file",
    )
    parser.add_argument("--target", help="Target column name")
    parser.add_argument("--submission-file", default="submission.csv", help="Submission CSV filename")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model name")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max iterative decisions")
    parser.add_argument("--force-tune", action="store_true", help="Force hyperparameter tuning of best model")
    parser.add_argument(
        "--metric-objective",
        default="auto",
        choices=["auto", "rmsle", "mse", "mae", "accuracy"],
        help="Optimization metric objective for model selection and tuning",
    )
    parser.add_argument("--show-langgraph", action="store_true", help="Display full LangGraph nodes, edges, and Mermaid")
    return parser.parse_args()


def ensure_api_key() -> str:
    load_dotenv()

    existing = os.getenv("OPENAI_API_KEY")
    if existing:
        return existing

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        set_openrouter_api_key(openrouter_key)
        return openrouter_key

    api_key = getpass.getpass("Enter your OpenRouter API key: ").strip()
    if not api_key:
        raise ValueError("OpenRouter API key is required")
    set_openrouter_api_key(api_key)
    return api_key


def validate_llm_connection(llm) -> None:
    """Fail fast when API key/model/base_url are invalid."""
    llm.invoke("Return exactly this token: READY")


def infer_target_column(file_path: str, sample_submission_path: str | None) -> str:
    if sample_submission_path and os.path.exists(sample_submission_path):
        with open(sample_submission_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])

        sample_columns = [str(col).strip() for col in header if str(col).strip()]
        if len(sample_columns) >= 2:
            target = sample_columns[-1]
            print(f"[main.py] Auto-detected target from sample submission: {target}")
            return target
        if len(sample_columns) == 1 and sample_columns[0].lower() != "id":
            target = sample_columns[0]
            print(f"[main.py] Auto-detected target from sample submission: {target}")
            return target

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = [str(col).strip() for col in next(reader, []) if str(col).strip()]

    if not columns:
        raise ValueError(f"Unable to detect target from empty header in {file_path}")

    preferred_targets = ["Calories", "diagnosed_diabetes", "target", "label", "class"]
    for candidate in preferred_targets:
        if candidate in columns:
            print(f"[main.py] Auto-detected target from train header: {candidate}")
            return candidate

    target = columns[-1]
    print(f"[main.py] Auto-detected target from train header: {target}")
    return target


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        ensure_api_key()
        llm = build_llm(model_name=args.model)
        validate_llm_connection(llm)
        target_column = args.target or infer_target_column(args.file, args.sample_submission)
        final_state = run_agent(
            file_path=args.file,
            test_file_path=args.test_file,
            sample_submission_file_path=args.sample_submission,
            competition_description_path=args.competition_description,
            submission_file_name=args.submission_file,
            target_column=target_column,
            llm=llm,
            max_iterations=args.max_iterations,
            force_tune=args.force_tune,
            metric_objective=args.metric_objective,
            show_langgraph=args.show_langgraph,
        )
        print_final_report(final_state)
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
