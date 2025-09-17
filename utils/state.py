# state.py
import pandas as pd
from typing import Dict, List, Any, Optional

"""
Global state management for the data analysis workflow.
"""

class WorkflowState:
    def __init__(self):
        self.datasets = {}        # dataset_name -> DataFrame
        self.profiles = {}        # dataset_name -> profile_dict
        self.analysis = {}        # dataset_name -> analysis_results
        self.questions = {}       # dataset_name -> list_of_questions
        self.understanding = {}   # dataset_name -> understanding_dict
        self.insights = {}        # dataset_name -> list_of_insights
        self.agents = {}          # NEW: agent_name -> agent_instance
        self.active_dataset: Optional[str] = None   # NEW

    def clear(self):
        """Clear all stored data."""
        self.datasets.clear()
        self.profiles.clear()
        self.analysis.clear()
        self.questions.clear()
        self.understanding.clear()
        self.insights.clear()
        self.agents.clear()  # NEW
    
    def set_active_dataset(self, dataset_name: str):
        """Mark a dataset as active for context-based questions."""
        if dataset_name in self.datasets:
            self.active_dataset = dataset_name
            print(f"✅ Successfully set active dataset: {dataset_name}")
        else:
            print(f"⚠️ Warning: Dataset '{dataset_name}' not found in STATE.datasets")
            print(f"🔍 Available datasets: {list(self.datasets.keys())}")
            # Set it anyway - it might be added later
            self.active_dataset = dataset_name
            print(f"ℹ️ Set as active anyway - will work once dataset is properly registered")
    
    def get_agent(self, agent_name: str = "insight"):
        """Get or create an agent instance."""
        if agent_name not in self.agents:
            if agent_name == "insight":
                from nodes.visualizationinsightsnode import InsightAgent
                self.agents[agent_name] = InsightAgent()
            # Add other agents here as needed
        return self.agents[agent_name]
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """Get summary info about a dataset."""
        if dataset_name not in self.datasets:
            return {"error": f"Dataset '{dataset_name}' not found"}
        
        df = self.datasets[dataset_name]
        return {
            "name": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "has_profile": dataset_name in self.profiles,
            "has_questions": dataset_name in self.questions,
            "num_insights": len(self.insights.get(dataset_name, []))
        }
    
    def list_datasets(self) -> list:
        """List all loaded datasets."""
        return list(self.datasets.keys())

# Global state instance
STATE = WorkflowState()