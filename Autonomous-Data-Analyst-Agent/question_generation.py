import pandas as pd
import itertools
from typing import List, Dict
from hf_client import get_hf_client

class ProactiveQuestionGenerator:
    def __init__(self):
        self.client, self.model_name = get_hf_client()

    def _call_hf_model(self, prompt: str) -> str:
        """Call Hugging Face model for question generation"""
        if not self.client:
            return ""
        try:
            output = self.client.text_generation(
                model=self.model_name,
                inputs=prompt,
                max_new_tokens=200
            )
            if isinstance(output, list) and len(output) > 0:
                return output[0].get("generated_text", "")
            return ""
        except Exception as e:
            print(f"⚠️ Hugging Face model call failed: {e}")
            return ""

    def generate_numeric_questions(self, df: pd.DataFrame) -> List[str]:
        """Generate questions for numeric columns"""
        questions = []
        numerics = df.select_dtypes(include=["number"]).columns.tolist()
        for col in numerics:
            questions.append(f"What is the distribution of {col}?")
            questions.append(f"What is the mean, median, and standard deviation of {col}?")
        # Pairwise correlations
        for col1, col2 in itertools.combinations(numerics, 2):
            questions.append(f"Is there a correlation between {col1} and {col2}?")
        return questions

    def generate_categorical_questions(self, df: pd.DataFrame) -> List[str]:
        """Generate questions for categorical columns"""
        questions = []
        categoricals = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categoricals:
            questions.append(f"What are the top values for {col}?")
            questions.append(f"What is the frequency distribution of {col}?")
        return questions

    def generate_text_questions(self, df: pd.DataFrame, text_columns: List[str]) -> List[str]:
        """Generate questions for text columns using HF LLM"""
        questions = []
        if not text_columns:
            return questions

        for col in text_columns:
            docs = df[col].dropna().astype(str).tolist()
            sample_docs = docs[:5]  # first 5 docs as sample
            prompt = (
                f"You are a data analyst. The dataset column '{col}' contains text data. "
                f"Here are some sample entries: {sample_docs}. "
                f"Generate 3 insightful analytical questions a data analyst might ask about this column."
            )
            hf_output = self._call_hf_model(prompt)
            
            # Attempt to split output into individual questions
            generated_questions = [q.strip("- ").strip() for q in hf_output.split("\n") if q.strip()]
            questions.extend(generated_questions[:3])  # take top 3
        return questions

    def generate_all_questions(self, df: pd.DataFrame, text_columns: List[str] = None) -> List[str]:
        """Generate proactive questions for the entire dataset"""
        questions = []
        questions += self.generate_numeric_questions(df)
        questions += self.generate_categorical_questions(df)
        if text_columns:
            questions += self.generate_text_questions(df, text_columns)
        return questions

    # ----------------- Run method for LangGraph -----------------
    def run(self, df: pd.DataFrame, text_columns: List[str] = None) -> List[str]:
        """
        Entry point for LangGraph workflow.
        Takes a cleaned DataFrame and optional text columns.
        Returns a list of proactive questions.
        """
        if df is None or df.empty:
            print("⚠️ Empty DataFrame provided to ProactiveQuestionGenerator.run()")
            return []
        questions = self.generate_all_questions(df, text_columns=text_columns)
        return questions

