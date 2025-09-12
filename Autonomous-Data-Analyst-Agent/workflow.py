# workflow.py
from file_ingestor import FileIngestor
from data_cleaning_analysis import DataCleaningNode  # or CleaningNode wrapper around your cleaning functions
from question_generation import ProactiveQuestionGenerator
from insights_generation import ProactiveInsightAgent
from report_generation import ProactiveReportGeneratorPDF
from chat_agent import ConversationalAgent  # your chat node implementation
from state import ReportState

def main():
    # Initialize state and agents
    state = FileIngestor()  # holds datasets & vector stores
    cleaning_node = DataCleaningNode()
    question_agent = ProactiveQuestionGenerator()
    insight_agent = ProactiveInsightAgent()
    report_agent = ProactiveReportGeneratorPDF()
    chat_agent = ConversationalAgent()

    # 1️⃣ Data Ingestion
    file_path = input("Enter file path (CSV, Excel, JSON, PDF, TXT): ")
    dataset_name = input("Enter a name for this dataset: ")
    df = state.ingest_file(dataset_name, file_path)

    # 2️⃣ Data Cleaning & Analysis
    state.dataset = df
    df_clean = cleaning_node.run(df)  # run method cleans & updates df
    state.dataset = df_clean  # update cleaned dataset

    # 3️⃣ Proactive Question Generation
    text_columns = [col for col in df_clean.columns if df_clean[col].dtype == "object"]
    questions = question_agent.generate_all_questions(df_clean, text_columns)
    state.questions = questions
    print("\nGenerated Questions:")
    for q in questions[:10]:  # show top 10
        print("-", q)

    # 4️⃣ Insight Generation
    insights = insight_agent.generate_proactive_insights(df_clean, text_columns)
    state.insights = insights
    print(f"\nTop Insights Generated: {len(insights)}")

    # 5️⃣ PDF Report Generation
    report_file = report_agent.generate_report(insights, filename="Proactive_Report.pdf")
    print(f"\n✅ PDF report generated: {report_file}")

    # 6️⃣ Chat Loop
    print("\nYou can now ask questions about the dataset. Type 'exit' to quit.")
    while True:
        q = input("Ask a question: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = chat_agent.ask_question(q, state)
        print(f"💬 Answer: {answer}")

if __name__ == "__main__":
    main()