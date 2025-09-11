# ---------------- workflow.py ----------------
import os
from typing import TypedDict
from state import ReportState
from dataingestionnode import data_ingestion_node
from dataanalysisnode import data_cleaning_analysis_node
from questiongenerationnode import (
    data_understanding_node,
    data_question_generation_node
)
from conversationalagent import ConversationalAgent
from visualizationinsightsnode import InsightAgent
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

# ---------------- Main ----------------
def main():
    dataset_name = input("Enter a name for your dataset: ").strip()
    file_path = input("Enter the full path of the dataset file: ").strip()

    file_type = detect_file_type(file_path)
    if not file_type:
        print("❌ Could not detect file type from extension.")
        print("Please ensure the file has a supported extension (csv, xlsx, json, txt, pdf, image formats).")
        return

    print(f"✅ Detected file type: {file_type}")

    state = ReportState()
    graph = create_graph()

    inputs = {
        "state": state,
        "dataset_name": dataset_name,
        "file_path": file_path,
        "file_type": file_type,
        "selected_question": ""
    }
    result = graph.invoke(inputs)

    print("\n✅ Workflow completed up to question generation!")
    print("Available datasets:", list(result["state"]["processed_tables"].keys()))

    questions = result["state"].get("generated_questions", {}).get(dataset_name, [])
    print("\n❓ Suggested Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    if not questions:
        print("⚠️ No questions generated. You can still ask your own.\n")

    ingestor = result["state"]["file_ingestor"]
    #conv_agent = ConversationalAgent(ingestor)
    conv_agent = ConversationalAgent(result["state"])
    insight_agent = InsightAgent(model="gemini-2.5-flash")

    # Update the chat loop in workflow.py main() function
    print("\n💬 Chat mode started! Ask questions about your dataset.")
    print("👉 You can either:")
    print("   - Type a free question")
    print("   - Enter the number of a suggested question")
    print("   - Type 'history' or 'show history' to see previous messages")
    print("   - Type 'visualization' or 'viz' to see the last visualization")
    print("   - Type 'exit' to quit\n")

    last_visualization = None

    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("👋 Ending chat.")
            break

        # Replace the history section in the chat loop with:
        if user_q.lower() in {"history", "show history", "chat history"}:
            history_text = conv_agent.get_chat_history_formatted()
            print(f"\n📜 Chat History:\n{history_text}\n")
            continue

        if user_q.lower() in {"visualization", "viz"}:
            if last_visualization:
                print("\n📊 Last Visualization:")
                print("(HTML content available for rendering in frontend)")
                print(f"Visualization type generated for your last question")
            else:
                print("⚠️ No visualization generated yet. Ask a question first.")
            continue

        if user_q.isdigit():
            q_idx = int(user_q) - 1
            if 0 <= q_idx < len(questions):
                user_q = questions[q_idx]
                print(f"👉 Using suggested question: {user_q}")
            else:
                print("⚠️ Invalid number. Please choose a valid question index.")
                continue

        try:
            # Get answer from conversational agent (vector store)
            vector_answer = conv_agent.ask(dataset_name, user_q)
            
            # Get insights and visualization from InsightAgent
            df = ingestor.get_dataset(dataset_name)
            insight_result = insight_agent.answer(df, user_q)
            insight_answer = insight_result.get("answer", "")
            viz_html = insight_result.get("visualization_html")
            
            # Store the last visualization
            last_visualization = viz_html
            
            # Combine both answers for a comprehensive response
            print(f"\n🤖 Agent Response:")
            print(f"📚 From dataset context: {vector_answer}")
            
            if insight_answer and insight_answer != vector_answer:
                print(f"📊 Data insights: {insight_answer}")
            
            if viz_html:
                print("📈 Visualization generated! Type 'viz' to see details or render the HTML in your application.")
            else:
                print("ℹ️  No visualization was generated for this question.")
            
            print()

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()