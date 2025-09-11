
# workflow.py
import os
from typing import TypedDict
from state import ReportState
from dataingestionnode import data_ingestion_node
from dataanalysisnode import data_cleaning_analysis_node
from questiongenerationnode import (
    data_understanding_node,
    data_question_generation_node
)
from visualizationinsightsnode import InsightAgent
from langgraph.graph import StateGraph
from hypothesisnode import HypothesisTesterLLM


# ---------------- Workflow State Schema ----------------
class WorkflowState(TypedDict):
    state: ReportState
    dataset_name: str
    file_path: str
    file_type: str
    selected_question: str

# ---------------- Wrappers ----------------
def ingestion_wrapper(workflow_state: WorkflowState):
    new_state = data_ingestion_node(
        workflow_state["state"],
        workflow_state["dataset_name"],
        workflow_state["file_path"],
        workflow_state["file_type"],
    )
    return {"state": new_state}
def hypothesis_testing_wrapper(workflow_state: WorkflowState):
    state = workflow_state["state"]
    dataset_name = workflow_state["dataset_name"]

    if "processed_tables" not in state or dataset_name not in state["processed_tables"]:
        return {"state": state}

    df = state["processed_tables"][dataset_name]
    tester = HypothesisTesterLLM()

    # Step 1: Generate hypotheses
    hyps = tester.generate_hypotheses(df)

    # Step 2: Test each hypothesis
    results = []
    for h in hyps:
        res = tester.test_hypothesis(df, h)
        results.append(res)

    # Rank results by score
    ranked = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    # Store in state
    state_insights = state.get("insights", {})
    if not isinstance(state_insights, dict):
        state_insights = {}
    state_insights[dataset_name] = {
        "hypotheses": hyps,
        "tested_insights": ranked,
    }
    state["insights"] = state_insights

    return {"state": state}

def analysis_wrapper(workflow_state: WorkflowState):
    new_state = data_cleaning_analysis_node(workflow_state["state"])
    return {"state": new_state}


def understanding_wrapper(workflow_state: WorkflowState):
    new_state = data_understanding_node(workflow_state["state"])
    return {"state": new_state}


def question_generation_wrapper(workflow_state: WorkflowState):
    new_state = data_question_generation_node(workflow_state["state"], num_questions=20)
    return {"state": new_state}


def exploration_loop_wrapper(workflow_state: WorkflowState):
    """
    Run autonomous exploration over generated questions/hypotheses.
    For each question, call the InsightAgent, collect answers and scores.
    Store a ranked list of insights in state.
    """
    state = workflow_state["state"]
    dataset_name = workflow_state["dataset_name"]

    if "processed_tables" not in state or dataset_name not in state["processed_tables"]:
        return {"state": state}

    df = state["processed_tables"][dataset_name]

    # Obtain candidate questions (from previous node) or generate fallback set
    gen_q = state.get("generated_questions", {}) or {}
    candidates = gen_q.get(dataset_name, []) if isinstance(gen_q, dict) else []

    # If still empty, provide a few heuristic hypotheses
    if not candidates:
        candidates = [
            "Provide a high-level overview of this dataset.",
            "Which numeric features correlate most strongly with each other?",
            "Are there any strong trends over time in the dataset?",
            "Which categorical groups differ most on the target variable?",
            "Detect top anomalies or outliers in the data."
        ]

    agent = InsightAgent()
    all_results = []

    for q in candidates:
        try:
            res = agent.answer(df, q)
            # Score the result heuristically (InsightAgent provides scoring helper)
            score = getattr(agent, "score_insight", None)
            if callable(score):
                s = score(df, res, q)
            else:
                # default simple scoring: prefer answers that include correlations or numeric summaries
                s = 0
                if isinstance(res, dict):
                    text = (res.get("answer") or "")
                    if "correl" in text.lower():
                        s += 2
                    if "trend" in text.lower() or "r^2" in text.lower() or "r2" in text.lower():
                        s += 1.5
                    if res.get("visualization_html"):
                        s += 0.5
            all_results.append({"question": q, "answer": res.get("answer"), "viz_html": res.get("visualization_html"), "score": s})
        except Exception as e:
            all_results.append({"question": q, "answer": f"Error running analysis: {e}", "viz_html": None, "score": 0})

    # Rank insights
    ranked = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)

    # Keep top-N (configurable)
    top_n = ranked[:6]

    # Produce a short summary (could call an LLM - but keep fallback deterministic)
    summary_lines = [f"Top {len(top_n)} insights:"]
    for i, item in enumerate(top_n, 1):
        summary_lines.append(f"{i}. {item['question']} (score: {item.get('score',0)})")

    summary_report = "\n".join(summary_lines)

    # Store in state
    state_insights = state.get("insights", {})
    if not isinstance(state_insights, dict):
        state_insights = {}
    state_insights[dataset_name] = {"ranked_insights": top_n, "summary_report": summary_report}
    state["insights"] = state_insights

    return {"state": state}


# ---------------- Graph Definition ----------------
def create_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("DataIngestion", ingestion_wrapper)
    graph.add_node("DataCleaningAndAnalysis", analysis_wrapper)
    graph.add_node("DataUnderstanding", understanding_wrapper)
    graph.add_node("QuestionGeneration", question_generation_wrapper)
    graph.add_node("ExplorationLoop", exploration_loop_wrapper)
    graph.add_node("HypothesisTesting", hypothesis_testing_wrapper)  # ✅ add this

    graph.add_edge("DataIngestion", "DataCleaningAndAnalysis")
    graph.add_edge("DataCleaningAndAnalysis", "DataUnderstanding")
    graph.add_edge("DataUnderstanding", "QuestionGeneration")
    graph.add_edge("QuestionGeneration", "ExplorationLoop")
    graph.add_edge("QuestionGeneration", "HypothesisTesting")         # ✅ link here
    graph.add_edge("HypothesisTesting", "ExplorationLoop")

    graph.set_entry_point("DataIngestion")

    return graph.compile()


# ---------------- Auto-detect File Type ----------------
def detect_file_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".csv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
        ".txt": "text",
        ".pdf": "pdf",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".bmp": "image",
        ".tiff": "image"
    }
    return mapping.get(ext, "")


if __name__ == "__main__":
    pass


## visualizationinsightsnode.py
# visualizationinsightsnode.py



## app.py

# app.py
