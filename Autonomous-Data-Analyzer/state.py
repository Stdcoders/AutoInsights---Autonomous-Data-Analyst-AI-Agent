from typing import TypedDict, Dict, Any, List
import pandas as pd

class ReportState:
    """
    This class represents the state of the report generation process.
    It contains all the necessary information to generate a report.
    """

    def __init__(self, input_context: Dict[str, Any] = None):
        input_context = input_context or {}
        self.llm_model = input_context.get("llm_model", None)
        self.processed_tables = input_context.get("processed_tables", {})
        self.pdf_path = input_context.get("pdf_path", "")
        self.analytics_plan = []
        self.query_results = []
        self.visualizations = []
        self.report_content = []
        self.analytics_code = []

    # 👇 Add these so you can use dict-like access in nodes
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)
    def get(self, key, default=None):   
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ReportState to a dictionary."""
        return {
            "llm_model": self.llm_model,
            "processed_tables": self.processed_tables,
            "analytics_plan": self.analytics_plan,
            "query_results": self.query_results,
            "visualizations": self.visualizations,
            "report_content": self.report_content,
            "pdf_path": self.pdf_path,
        }


