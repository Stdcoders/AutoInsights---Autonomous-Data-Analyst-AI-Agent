import os
from typing import TypedDict
from state import ReportState
from dataingestionnode import data_ingestion_node
from dataanalysisnode import data_cleaning_analysis_node
from questiongenerationnode import (
    data_understanding_node,
    data_question_generation_node,
)
from visualizationinsightsnode import InsightAgent  # <-- new agent with visualizations
from langgraph.graph import StateGraph


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


def analysis_wrapper(workflow_state: WorkflowState):
    new_state = data_cleaning_analysis_node(workflow_state["state"])
    return {"state": new_state}


def understanding_wrapper(workflow_state: WorkflowState):
    new_state = data_understanding_node(workflow_state["state"])
    return {"state": new_state}


def question_generation_wrapper(workflow_state: WorkflowState):
    new_state = data_question_generation_node(workflow_state["state"], num_questions=10)
    return {"state": new_state}


def insight_wrapper(workflow_state: WorkflowState):
    state = workflow_state["state"]
    dataset_name = workflow_state["dataset_name"]
    selected_question = workflow_state.get("selected_question")

    if not selected_question:
        raise ValueError("❌ No question selected for insight generation.")

    df = state["processed_tables"][dataset_name]
    agent = InsightAgent()
    result = agent.answer(df, selected_question)

    state["insights"] = {dataset_name: result}
    return {"state": state}


# ---------------- Graph Definition ----------------
def create_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("DataIngestion", ingestion_wrapper)
    graph.add_node("DataCleaningAndAnalysis", analysis_wrapper)
    graph.add_node("DataUnderstanding", understanding_wrapper)
    graph.add_node("QuestionGeneration", question_generation_wrapper)

    # ✅ REMOVE DataInsight from the graph
    # graph.add_node("DataInsight", insight_wrapper)

    graph.add_edge("DataIngestion", "DataCleaningAndAnalysis")
    graph.add_edge("DataCleaningAndAnalysis", "DataUnderstanding")
    graph.add_edge("DataUnderstanding", "QuestionGeneration")

    graph.set_entry_point("DataIngestion")

    return graph.compile()


# ---------------- Main ----------------
import webbrowser

def main():
    # User inputs
    dataset_name = input("Enter a name for your dataset: ").strip()
    file_path = input("Enter the full path of the dataset file: ").strip()
    file_type = input("Enter file type (csv/excel/json/text/pdf/image): ").strip().lower()

    # Initialize empty state
    state = ReportState()

    # Build workflow graph
    graph = create_graph()

    # Run workflow until QuestionGeneration
    inputs = {
        "state": state,
        "dataset_name": dataset_name,
        "file_path": file_path,
        "file_type": file_type,
    }
    result = graph.invoke(inputs)

    print("\n✅ Workflow completed up to question generation!")
    print("Available datasets:", list(result["state"]["processed_tables"].keys()))

    # Show generated questions
    questions = result["state"].get("generated_questions", {}).get(dataset_name, [])
    print("\n❓ Suggested Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    if not questions:
        print("⚠️ No questions generated.")
        return

    # Let user pick a question
    choice = int(input("\nSelect a question number for insight (e.g., 1): ").strip())
    selected_question = questions[choice - 1]

    print(f"\n👉 You selected: {selected_question}")

    # === Run InsightAgent outside graph ===
    from visualizationinsightsnode import InsightAgent
    agent = InsightAgent()
    df = result["state"]["processed_tables"][dataset_name]

    insight = agent.answer(df, selected_question)

    print("\n📊 Insight Answer:")
    print(insight["answer"])

    # Show visualization directly
    if insight["visualization_html"]:
        output_file = f"{dataset_name}_insight.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(insight["visualization_html"])
        webbrowser.open(f"file://{os.path.abspath(output_file)}")  # <-- auto open
        print("✅ Visualization opened in your browser.")

if __name__ == "__main__":
    main()