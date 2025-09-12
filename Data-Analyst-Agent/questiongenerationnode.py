import os
import json
import re
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from state import ReportState

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment. Please set it in your .env file.")

genai.configure(api_key=API_KEY)
'''def format_data_understanding(understanding: dict) -> str:
    lines = []
    domain = understanding.get("domain", "Unknown")
    column_roles = understanding.get("column_roles", {})
    use_cases = understanding.get("use_cases", [])
    limitations = understanding.get("limitations", [])

    lines.append(f"📂 **Domain**: {domain}\n")

    lines.append("📊 **Column Roles:**")
    for col, role in column_roles.items():
        lines.append(f"- **{col}**: {role}")
    lines.append("")

    lines.append("💡 **Use Cases:**")
    for idx, case in enumerate(use_cases, 1):
        lines.append(f"{idx}. {case}")
    lines.append("")

    lines.append("⚠️ **Potential Limitations or Biases:**")
    for idx, limitation in enumerate(limitations, 1):
        lines.append(f"{idx}. {limitation}")
    lines.append("")

    return "\n".join(lines)'''
def format_data_understanding(understanding: dict) -> str:
    """Convert JSON understanding to user-friendly text format"""
    lines = []
    
    # Handle case where understanding might be raw text
    if isinstance(understanding, str) or "raw_text" in understanding:
        raw_text = understanding if isinstance(understanding, str) else understanding.get("raw_text", "")
        return f"📋 Raw Analysis:\n{raw_text}"
    
    domain = understanding.get("domain", "Unknown")
    column_roles = understanding.get("column_roles", {})
    use_cases = understanding.get("use_cases", [])
    limitations = understanding.get("limitations", [])
    additional_insights = understanding.get("additional_insights", [])

    lines.append(f"🏷️  **Domain Identification**: {domain}\n")

    if column_roles:
        lines.append("📊 **Column Analysis:**")
        for col, role in column_roles.items():
            lines.append(f"   • **{col}**: {role}")
        lines.append("")

    if use_cases:
        lines.append("💡 **Potential Use Cases:**")
        for idx, case in enumerate(use_cases, 1):
            lines.append(f"   {idx}. {case}")
        lines.append("")

    if limitations:
        lines.append("⚠️  **Limitations & Considerations:**")
        for idx, limitation in enumerate(limitations, 1):
            lines.append(f"   {idx}. {limitation}")
        lines.append("")

    if additional_insights:
        lines.append("🔍 **Additional Insights:**")
        for idx, insight in enumerate(additional_insights, 1):
            lines.append(f"   {idx}. {insight}")

    return "\n".join(lines)

'''def data_understanding_node(state: ReportState, model: str = "gemini-1.5-flash") -> ReportState:
    """
    Uses LLM to semantically understand the dataset domain and context.
    """
    if "processed_tables" not in state or "file_ingestor" not in state:
        raise ValueError("No ingested data found in state.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found in environment.")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model)

    state["data_understanding"] = {}

    for dataset_name in state["processed_tables"].keys():
        profile = state["file_ingestor"].get_profile(dataset_name)

        prompt = f"""
        You are a domain expert. I have a dataset with the following profile:

        {json.dumps(profile, indent=2, default=str)}

        Please provide a structured analysis:
        1. What domain (e.g. healthcare, finance, retail, etc.) does this dataset most likely belong to?
        2. Interpret the role of each column in plain English (what it represents, why it may be important).
        3. Suggest 3–5 real-world use cases for analyzing this dataset.
        4. Mention any potential limitations or biases.

        Return your response as JSON with keys: domain, column_roles, use_cases, limitations.
        """

        response = llm.generate_content(prompt, generation_config={"temperature": 0.4, "max_output_tokens": 600})
        response_text = response.candidates[0].content.parts[0].text

        try:
            understanding = json.loads(response_text)
        except Exception:
            understanding = {"raw_text": response_text}  # fallback

        state["data_understanding"][dataset_name] = understanding

        print(f"\n🤖 Data Understanding (LLM) for '{dataset_name}':")
        print(json.dumps(understanding, indent=2, default=str))

    return state'''

def data_understanding_node(state: ReportState, model: str = "gemini-1.5-flash") -> ReportState:
    """Uses LLM to semantically understand the dataset domain and context."""
    if "processed_tables" not in state or "file_ingestor" not in state:
        raise ValueError("No ingested data found in state.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found in environment.")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model)

    state["data_understanding"] = {}
    state["data_understanding_formatted"] = {}  # Add formatted version

    for dataset_name in state["processed_tables"].keys():
        profile = state["file_ingestor"].get_profile(dataset_name)

        prompt = f"""
        You are a domain expert. I have a dataset with the following profile:

        {json.dumps(profile, indent=2, default=str)}

        Please provide a structured analysis:
        1. What domain (e.g. healthcare, finance, retail, etc.) does this dataset most likely belong to?
        2. Interpret the role of each column in plain English (what it represents, why it may be important).
        3. Suggest 3–5 real-world use cases for analyzing this dataset.
        4. Mention any potential limitations or biases.

        Return your response as JSON with keys: domain, column_roles, use_cases, limitations.
        """

        response = llm.generate_content(prompt, generation_config={"temperature": 0.4, "max_output_tokens": 600})
        response_text = response.candidates[0].content.parts[0].text

        try:
            understanding = json.loads(response_text)
        except Exception:
            understanding = {"raw_text": response_text}  # fallback

        state["data_understanding"][dataset_name] = understanding
        
        # Create user-friendly formatted version
        formatted_understanding = format_data_understanding(understanding)
        state["data_understanding_formatted"][dataset_name] = formatted_understanding

        print(f"\n🤖 Data Understanding for '{dataset_name}':")
        print("=" * 50)
        print(formatted_understanding)
        print("=" * 50)

    return state



class QuestionGenerationAgent:
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model)

    @staticmethod
    def _json_serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

    def generate(self, profile, understanding, num_questions: int = 10):
        prompt = f"""
        You are a senior data analyst. Based on the dataset profile and domain understanding below,
        generate {num_questions} insightful analytical questions.

        Dataset profile:
        {json.dumps(profile, indent=2, default=self._json_serialize)}

        Dataset understanding:
        {json.dumps(understanding, indent=2, default=self._json_serialize)}

        Generate questions that are:
        - Relevant to the dataset columns, types, and statistics
        - Domain-aware (use the understanding context)
        - Answerable using data analysis, visualization, or modeling

        Return ONLY a valid JSON array of strings like:
        ["question 1", "question 2", ...]
        """

        response = self.model.generate_content(prompt, generation_config={"temperature": 0.3, "max_output_tokens": 500})
        response_text = response.candidates[0].content.parts[0].text

        try:
            match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        # fallback if parsing fails
        cols = [col["name"] for col in profile.get("columns", []) if isinstance(col, dict)]
        return [f"What trends can we analyze in {col}?" for col in cols[:num_questions]]


def data_question_generation_node(state: ReportState, num_questions: int = 10) -> ReportState:
    """
    Node to generate analytical questions using both dataset profile + understanding.
    """
    if "processed_tables" not in state or "file_ingestor" not in state:
        raise ValueError("No ingested data found in state.")
    if "data_understanding" not in state:
        raise ValueError("Run data_understanding_node before generating questions.")

    agent = QuestionGenerationAgent()
    state["generated_questions"] = {}

    for dataset_name in state["processed_tables"].keys():
        profile = state["file_ingestor"].get_profile(dataset_name)
        understanding = state["data_understanding"].get(dataset_name, {})

        questions = agent.generate(profile, understanding, num_questions)
        state["generated_questions"][dataset_name] = questions

        print(f"\n❓ Generated Questions for '{dataset_name}':")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}")

    return state
