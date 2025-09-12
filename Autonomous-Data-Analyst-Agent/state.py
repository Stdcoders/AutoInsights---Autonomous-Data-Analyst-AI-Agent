from typing import Dict, List
import pandas as pd

class ReportState:
    def __init__(self):
        # Raw and processed datasets
        self.datasets: Dict[str, pd.DataFrame] = {}       # Original ingested datasets
        self.cleaned_tables: Dict[str, pd.DataFrame] = {} # Cleaned / processed datasets
        self.analysis_results: Dict[str, dict] = {}       # Summary / EDA results per dataset

        # Proactive workflow
        self.questions: List[str] = []    # Generated analytical questions
        self.insights: List[dict] = []    # Generated insights (answers + visualizations)

    # ---------- Dataset Management ----------
    def add_dataset(self, name: str, df: pd.DataFrame):
        self.datasets[name] = df

    def get_dataset(self, name: str) -> pd.DataFrame:
        return self.datasets.get(name, None)

    def update_cleaned_table(self, name: str, df: pd.DataFrame):
        self.cleaned_tables[name] = df

    def get_cleaned_table(self, name: str) -> pd.DataFrame:
        return self.cleaned_tables.get(name, None)

    def update_analysis_result(self, name: str, analysis: dict):
        self.analysis_results[name] = analysis

    def get_analysis_result(self, name: str) -> dict:
        return self.analysis_results.get(name, {})

    # ---------- Questions & Insights ----------
    def update_questions(self, questions: List[str]):
        self.questions = questions

    def append_question(self, question: str):
        self.questions.append(question)

    def update_insights(self, insights: List[dict]):
        self.insights = insights

    def append_insight(self, insight: dict):
        self.insights.append(insight)

    # ---------- Convenience ----------
    def reset(self):
        """Reset all stored datasets, questions, and insights."""
        self.datasets = {}
        self.cleaned_tables = {}
        self.analysis_results = {}
        self.questions = []
        self.insights = []
