from hf_client import get_hf_client
from state import ReportState
from insights_generation import ProactiveInsightAgent
from insights_generation import plot_scatter, plot_histogram, plot_bar, plot_correlation_matrix

class ConversationalAgent:
    def __init__(self, hf_client: get_hf_client, report_state: ReportState):
        self.hf = hf_client
        self.state = report_state
        self.insight_agent = ProactiveInsightAgent(hf_client)

    def ask_question(self, question: str) -> dict:
        """
        Returns a dictionary with answer and optional visualization_html.
        """
        if self.state.cleaned_df is None:
            return {"answer": "No dataset loaded yet.", "visualization_html": None}

        # Step 1: Check if question relates to existing insights
        relevant_insights = [
            ins for ins in self.state.insights
            if any(col.lower() in question.lower() for col in str(ins["answer"]).lower().split())
        ]

        if relevant_insights:
            # Return top relevant insight + visualization
            response = relevant_insights[0]["answer"]
            viz_html = relevant_insights[0].get("visualization_html")
            return {"answer": response, "visualization_html": viz_html}

        # Step 2: Use HF LLM to answer new question
        df_sample = self.state.cleaned_df.head(5).to_dict(orient="records")
        cols = list(self.state.cleaned_df.columns)
        prompt = (
            f"You are a data analyst. Answer the user's question based on this dataset sample and columns:\n"
            f"Columns: {cols}\nSample: {df_sample}\nQuestion: {question}\n"
            "Return the answer in natural language. If needed, suggest a visualization type."
        )
        hf_response = self.hf.generate_text(prompt)

        # Optional: generate a visualization for numeric questions
        viz_html = None
        if "correlation" in question.lower() and len(cols) >= 2:
            viz_html = plot_correlation_matrix(self.state.cleaned_df)
        else:
            numeric_cols = self.state.cleaned_df.select_dtypes(include=["number"]).columns.tolist()
            if any(col.lower() in question.lower() for col in numeric_cols) and numeric_cols:
                viz_html = plot_histogram(self.state.cleaned_df, numeric_cols[0])

        # Log this question for future proactive analysis
        self.state.insights.append({"answer": hf_response, "visualization_html": viz_html, "score": 1})

        return {"answer": hf_response, "visualization_html": viz_html}

    def run(self, question: str) -> dict:
        """
        Wrapper for LangGraph execution or automated runs.
        """
        return self.ask_question(question)

