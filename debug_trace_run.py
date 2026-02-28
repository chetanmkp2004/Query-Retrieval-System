import traceback
from dotenv import load_dotenv
from config.settings import DEFAULT_MODEL, build_llm
from agent.runner import run_agent

load_dotenv()
llm = build_llm(model_name=DEFAULT_MODEL)

try:
    state = run_agent(
        file_path="data/train.csv",
        test_file_path="data/test.csv",
        sample_submission_file_path="data/sample_submission.csv",
        competition_description_path="data/data_description.txt",
        submission_file_name="submissions.csv",
        target_column="SalePrice",
        llm=llm,
        max_iterations=1,
    )
    print("OK", state.get("best_model_name"))
except Exception:
    traceback.print_exc()
    raise
