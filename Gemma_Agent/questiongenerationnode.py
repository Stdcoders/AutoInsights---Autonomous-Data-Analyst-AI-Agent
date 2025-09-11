'''import os
import json
import re
import numpy as np
from state import ReportState
from gemma_llm import GemmaLLM

def format_data_understanding(understanding: dict) -> str:
    """Convert JSON understanding to user-friendly text format"""
    lines = []
    
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

def data_understanding_node(state: ReportState, model: str = "google/gemma-7b-it") -> ReportState:
    """Uses LLM to semantically understand the dataset domain and context."""
    if "processed_tables" not in state or "file_ingestor" not in state:
        raise ValueError("No ingested data found in state.")

    llm = GemmaLLM(model_name=model)
    state["data_understanding"] = {}
    state["data_understanding_formatted"] = {}

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

        response_text = llm._call(prompt)

        try:
            understanding = json.loads(response_text)
        except Exception:
            understanding = {"raw_text": response_text}

        state["data_understanding"][dataset_name] = understanding
        formatted_understanding = format_data_understanding(understanding)
        state["data_understanding_formatted"][dataset_name] = formatted_understanding

        print(f"\n🤖 Data Understanding for '{dataset_name}':")
        print("=" * 50)
        print(formatted_understanding)
        print("=" * 50)

    return state

class QuestionGenerationAgent:
    def __init__(self, model: str = "google/gemma-7b-it"):
        self.llm = GemmaLLM(model_name=model)

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

        response_text = self.llm._call(prompt)

        try:
            match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

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

    return state'''

'''import os
import json
import re
import numpy as np
import pandas as pd
from state import ReportState
from gemma_llm import GemmaLLM

def create_profile_from_dataframe(df: pd.DataFrame, dataset_name: str) -> dict:
    """Create a comprehensive profile from a cleaned DataFrame"""
    profile = {
        "dataset_name": dataset_name,
        "shape": df.shape,
        "columns": [],
        "missing_values": {},
        "basic_statistics": {},
        "data_types": {},
        "unique_values": {}
    }
    
    for column in df.columns:
        dtype = str(df[column].dtype)
        profile["columns"].append({
            "name": column,
            "dtype": dtype,
            "unique_count": df[column].nunique(),
            "missing_count": df[column].isnull().sum()
        })
        
        profile["missing_values"][column] = df[column].isnull().sum()
        profile["data_types"][column] = dtype
        profile["unique_values"][column] = df[column].nunique()
        
        # Add basic statistics for numerical columns
        if df[column].dtype in ['int64', 'float64']:
            profile["basic_statistics"][column] = {
                "mean": df[column].mean(),
                "median": df[column].median(),
                "std": df[column].std(),
                "min": df[column].min(),
                "max": df[column].max(),
                "q1": df[column].quantile(0.25),
                "q3": df[column].quantile(0.75)
            }
        elif df[column].dtype == 'object':
            # For categorical columns, show value counts (top 5)
            value_counts = df[column].value_counts().head(5).to_dict()
            profile["basic_statistics"][column] = {
                "top_values": value_counts,
                "most_common": df[column].mode()[0] if not df[column].mode().empty else None
            }
    
    return profile

def format_data_understanding(understanding: dict) -> str:
    """Convert JSON understanding to user-friendly text format"""
    lines = []
    
    if isinstance(understanding, str) or "raw_text" in understanding:
        raw_text = understanding if isinstance(understanding, str) else understanding.get("raw_text", "")
        return f"📋 Raw Analysis:\n{raw_text}"
    
    domain = understanding.get("domain", "Unknown")
    column_roles = understanding.get("column_roles", {})
    use_cases = understanding.get("use_cases", [])
    limitations = understanding.get("limitations", [])
    additional_insights = understanding.get("additional_insights", [])
    data_quality = understanding.get("data_quality_assessment", {})

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

    if data_quality:
        lines.append("🔍 **Data Quality Assessment:**")
        for metric, value in data_quality.items():
            lines.append(f"   • {metric.replace('_', ' ').title()}: {value}")
        lines.append("")

    if additional_insights:
        lines.append("🌟 **Additional Insights:**")
        for idx, insight in enumerate(additional_insights, 1):
            lines.append(f"   {idx}. {insight}")

    return "\n".join(lines)

def data_understanding_node(state: ReportState, model: str = "google/gemma-7b-it") -> ReportState:
    """Uses LLM to semantically understand the dataset domain and context using CLEANED data."""
    # Check for cleaned tables first, fall back to processed if not available
    if "cleaned_tables" in state:
        tables_key = "cleaned_tables"
        print("✅ Using CLEANED data for understanding")
    elif "processed_tables" in state:
        tables_key = "processed_tables"
        print("⚠️  Using RAW data for understanding (cleaned data not available)")
    else:
        raise ValueError("No data found in state.")

    llm = GemmaLLM(model_name=model)
    state["data_understanding"] = {}
    state["data_understanding_formatted"] = {}
    state["data_profiles"] = {}  # Store created profiles

    for dataset_name, df in state[tables_key].items():
        print(f"\n🤖 Analyzing cleaned dataset: {dataset_name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Create profile from the actual cleaned data
        profile = create_profile_from_dataframe(df, dataset_name)
        state["data_profiles"][dataset_name] = profile

        prompt = f"""
        You are a domain expert data analyst. I have a CLEANED dataset with the following profile:
        {json.dumps(profile, indent=2, default=str)}
        
        Please provide a comprehensive structured analysis:
        
        1. **Domain Identification**: What specific domain does this dataset belong to? (e.g., healthcare-patient records, finance-transactions, retail-sales, etc.)
        
        2. **Column Roles**: For each column, explain:
           - What real-world concept it represents
           - Why it's important for analysis
           - Potential relationships with other columns
        
        3. **Use Cases**: Suggest 3-5 specific, actionable analytical use cases
        
        4. **Limitations**: Identify any data limitations, potential biases, or analysis constraints
        
        5. **Data Quality Assessment**: Comment on the overall data quality based on the profile
        
        6. **Additional Insights**: Any other important observations
        
        Return your response as JSON with these keys: domain, column_roles, use_cases, limitations, data_quality_assessment, additional_insights.
        """

        response_text = llm._call(prompt)

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                understanding = json.loads(json_match.group(0))
            else:
                understanding = {"raw_text": response_text}
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            understanding = {"raw_text": response_text}

        state["data_understanding"][dataset_name] = understanding
        formatted_understanding = format_data_understanding(understanding)
        state["data_understanding_formatted"][dataset_name] = formatted_understanding

        print(f"\n🤖 Data Understanding for '{dataset_name}':")
        print("=" * 60)
        print(formatted_understanding)
        print("=" * 60)

    return state

class QuestionGenerationAgent:
    def __init__(self, model: str = "google/gemma-7b-it"):
        self.llm = GemmaLLM(model_name=model)

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
        You are a senior data analyst. Based on the CLEANED dataset profile and domain understanding below,
        generate {num_questions} insightful, specific analytical questions that can be answered using this data.
        
        **CLEANED DATASET PROFILE:**
        {json.dumps(profile, indent=2, default=self._json_serialize)}
        
        **DOMAIN UNDERSTANDING:**
        {json.dumps(understanding, indent=2, default=self._json_serialize)}
        
        Generate questions that are:
        - **Specific** to this cleaned dataset's columns and statistics
        - **Domain-aware** (use the understanding context)
        - **Answerable** using data analysis, visualization, or modeling
        - **Actionable** for business or research decisions
        - **Varied** across different analytical techniques (trends, correlations, patterns, predictions)
        
        Return ONLY a valid JSON array of strings like:
        ["question 1", "question 2", ...]
        """

        response_text = self.llm._call(prompt)

        try:
            # Extract JSON array from response
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                questions = json.loads(match.group(0))
                # Ensure we have the right number of questions
                if len(questions) > num_questions:
                    questions = questions[:num_questions]
                return questions
        except Exception as e:
            print(f"Question generation JSON parsing failed: {e}")
        
        # Fallback: generate basic questions based on columns
        cols = [col for col in profile.get("columns", [])]
        return [f"What are the trends and patterns in {col}?" for col in cols[:num_questions]]

def data_question_generation_node(state: ReportState, num_questions: int = 10) -> ReportState:
    """
    Node to generate analytical questions using CLEANED dataset profile + understanding.
    """
    # Check dependencies
    if "cleaned_tables" not in state:
        raise ValueError("No cleaned data found. Run data_cleaning_analysis_node first.")
    
    if "data_understanding" not in state:
        raise ValueError("No data understanding found. Run data_understanding_node first.")
    
    if "data_profiles" not in state:
        raise ValueError("No data profiles found. Run data_understanding_node first.")

    agent = QuestionGenerationAgent()
    state["generated_questions"] = {}
    state["question_generation_metadata"] = {}

    for dataset_name in state["cleaned_tables"].keys():
        print(f"\n❓ Generating questions for cleaned dataset: {dataset_name}")
        
        profile = state["data_profiles"][dataset_name]
        understanding = state["data_understanding"].get(dataset_name, {})
        
        # Verify we have clean data
        df = state["cleaned_tables"][dataset_name]
        print(f"   Clean data shape: {df.shape}")
        print(f"   Available columns: {list(df.columns)}")

        questions = agent.generate(profile, understanding, num_questions)
        state["generated_questions"][dataset_name] = questions
        
        # Store metadata about the generation
        state["question_generation_metadata"][dataset_name] = {
            "source": "cleaned_data",
            "num_questions": len(questions),
            "dataset_shape": df.shape,
            "generation_timestamp": pd.Timestamp.now().isoformat()
        }

        print(f"\n❓ Generated {len(questions)} Questions for '{dataset_name}':")
        print("-" * 50)
        for i, q in enumerate(questions, 1):
            print(f"{i:2d}. {q}")
        print("-" * 50)

    return state'''
import os
import json
import re
import numpy as np
import pandas as pd
from state import ReportState
from gemma_llm import GemmaLLM

def create_profile_from_cleaned_data(df: pd.DataFrame, dataset_name: str, original_profile: dict) -> dict:
    """Create an enhanced profile from cleaned DataFrame, preserving original metadata"""
    profile = {
        "dataset_name": dataset_name,
        "shape": df.shape,
        "columns": [],
        "missing_values": {},
        "basic_statistics": {},
        "data_types": {},
        "unique_values": {},
        "cleaning_impact": {
            "rows_removed": original_profile.get('n_rows', 0) - df.shape[0],
            "columns_remaining": df.shape[1],
            "missing_values_removed": 0
        }
    }
    
    # Calculate missing values removed
    original_missing = sum(col.get('num_missing', 0) for col in original_profile.get('columns', []))
    current_missing = df.isnull().sum().sum()
    profile["cleaning_impact"]["missing_values_removed"] = original_missing - current_missing
    
    for column in df.columns:
        dtype = str(df[column].dtype)
        profile["columns"].append({
            "name": column,
            "dtype": dtype,
            "unique_count": df[column].nunique(),
            "missing_count": df[column].isnull().sum()
        })
        
        profile["missing_values"][column] = df[column].isnull().sum()
        profile["data_types"][column] = dtype
        profile["unique_values"][column] = df[column].nunique()
        
        # Add basic statistics for numerical columns
        if df[column].dtype in ['int64', 'float64']:
            profile["basic_statistics"][column] = {
                "mean": df[column].mean(),
                "median": df[column].median(),
                "std": df[column].std(),
                "min": df[column].min(),
                "max": df[column].max(),
                "q1": df[column].quantile(0.25),
                "q3": df[column].quantile(0.75)
            }
        elif df[column].dtype == 'object':
            # For categorical columns, show value counts (top 5)
            value_counts = df[column].value_counts().head(5).to_dict()
            profile["basic_statistics"][column] = {
                "top_values": value_counts,
                "most_common": df[column].mode()[0] if not df[column].mode().empty else None
            }
    
    return profile

def format_data_understanding(understanding: dict) -> str:
    """Convert JSON understanding to user-friendly text format"""
    lines = []
    
    if isinstance(understanding, str) or "raw_text" in understanding:
        raw_text = understanding if isinstance(understanding, str) else understanding.get("raw_text", "")
        return f"📋 Raw Analysis:\n{raw_text}"
    
    domain = understanding.get("domain", "Unknown")
    column_roles = understanding.get("column_roles", {})
    use_cases = understanding.get("use_cases", [])
    limitations = understanding.get("limitations", [])
    additional_insights = understanding.get("additional_insights", [])
    data_quality = understanding.get("data_quality_assessment", {})

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

    if data_quality:
        lines.append("🔍 **Data Quality Assessment:**")
        for metric, value in data_quality.items():
            lines.append(f"   • {metric.replace('_', ' ').title()}: {value}")
        lines.append("")

    if additional_insights:
        lines.append("🌟 **Additional Insights:**")
        for idx, insight in enumerate(additional_insights, 1):
            lines.append(f"   {idx}. {insight}")

    return "\n".join(lines)

def data_understanding_node(state: ReportState, model: str = "google/gemma-7b-it") -> ReportState:
    """Uses LLM to semantically understand the dataset domain and context using CLEANED data."""
    # Check for cleaned tables first
    if "cleaned_tables" not in state:
        raise ValueError("No cleaned data found. Run data_cleaning_analysis_node first.")
    
    if "file_ingestor" not in state:
        raise ValueError("No file ingestor found in state.")

    llm = GemmaLLM(model_name=model)
    state["data_understanding"] = {}
    state["data_understanding_formatted"] = {}
    state["data_profiles"] = {}  # Store enhanced profiles from cleaned data

    ingestor = state["file_ingestor"]

    for dataset_name, cleaned_df in state["cleaned_tables"].items():
        print(f"\n🤖 Analyzing cleaned dataset: {dataset_name}")
        print(f"   Cleaned shape: {cleaned_df.shape}")
        print(f"   Columns: {list(cleaned_df.columns)}")
        
        # Get original profile from ingestor
        try:
            original_profile = ingestor.get_profile(dataset_name)
        except KeyError:
            print(f"⚠️  Original profile not found for {dataset_name}, creating from cleaned data")
            original_profile = {"n_rows": cleaned_df.shape[0], "columns": []}
        
        # Create enhanced profile from cleaned data
        enhanced_profile = create_profile_from_cleaned_data(cleaned_df, dataset_name, original_profile)
        state["data_profiles"][dataset_name] = enhanced_profile

        prompt = f"""
        You are a domain expert data analyst. I have a CLEANED dataset with the following profile:
        
        **DATASET OVERVIEW:**
        - Name: {dataset_name}
        - Rows: {enhanced_profile['shape'][0]} (after cleaning)
        - Columns: {enhanced_profile['shape'][1]}
        - Rows removed during cleaning: {enhanced_profile['cleaning_impact']['rows_removed']}
        - Missing values removed: {enhanced_profile['cleaning_impact']['missing_values_removed']}
        
        **COLUMN DETAILS:**
        {json.dumps(enhanced_profile['columns'], indent=2, default=str)}
        
        **BASIC STATISTICS:**
        {json.dumps(enhanced_profile['basic_statistics'], indent=2, default=str)}
        
        Please provide a comprehensive structured analysis:
        
        1. **Domain Identification**: What specific domain does this CLEANED dataset belong to?
        
        2. **Column Roles**: For each column, explain what it represents and its importance
        
        3. **Use Cases**: Suggest 3-5 specific analytical use cases based on the cleaned data
        
        4. **Limitations**: Identify any remaining limitations or analysis constraints
        
        5. **Data Quality Assessment**: Comment on the current data quality post-cleaning
        
        6. **Additional Insights**: Any other important observations
        
        Return your response as JSON with these keys: domain, column_roles, use_cases, limitations, data_quality_assessment, additional_insights.
        """

        response_text = llm._call(prompt)

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                understanding = json.loads(json_match.group(0))
            else:
                understanding = {"raw_text": response_text}
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            understanding = {"raw_text": response_text}

        state["data_understanding"][dataset_name] = understanding
        formatted_understanding = format_data_understanding(understanding)
        state["data_understanding_formatted"][dataset_name] = formatted_understanding

        print(f"\n🤖 Data Understanding for '{dataset_name}':")
        print("=" * 60)
        print(formatted_understanding)
        print("=" * 60)

    return state

class QuestionGenerationAgent:
    def __init__(self, model: str = "google/gemma-7b-it"):
        self.llm = GemmaLLM(model_name=model)

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

    def generate(self, enhanced_profile, understanding, num_questions: int = 10):
        prompt = f"""
        You are a senior data analyst. Based on the CLEANED dataset profile and domain understanding below,
        generate {num_questions} insightful, specific analytical questions that can be answered using this cleaned data.
        
        **CLEANED DATASET PROFILE:**
        {json.dumps(enhanced_profile, indent=2, default=self._json_serialize)}
        
        **DOMAIN UNDERSTANDING:**
        {json.dumps(understanding, indent=2, default=self._json_serialize)}
        
        Generate questions that are:
        - **Specific** to this cleaned dataset's columns and statistics
        - **Domain-aware** (use the understanding context)
        - **Answerable** using the available cleaned data
        - **Actionable** for business or research decisions
        - **Varied** across different analytical techniques
        
        Return ONLY a valid JSON array of strings.
        """

        response_text = self.llm._call(prompt)

        try:
            # Extract JSON array from response
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                questions = json.loads(match.group(0))
                if len(questions) > num_questions:
                    questions = questions[:num_questions]
                return questions
        except Exception as e:
            print(f"Question generation JSON parsing failed: {e}")
        
        # Fallback: generate basic questions based on columns
        cols = [col["name"] for col in enhanced_profile.get("columns", [])]
        return [f"What are the trends and patterns in {col}?" for col in cols[:num_questions]]

def data_question_generation_node(state: ReportState, num_questions: int = 10) -> ReportState:
    """
    Node to generate analytical questions using CLEANED dataset profile + understanding.
    """
    # Check dependencies
    if "cleaned_tables" not in state:
        raise ValueError("No cleaned data found. Run data_cleaning_analysis_node first.")
    
    if "data_understanding" not in state:
        raise ValueError("No data understanding found. Run data_understanding_node first.")
    
    if "data_profiles" not in state:
        raise ValueError("No enhanced profiles found. Run data_understanding_node first.")

    agent = QuestionGenerationAgent()
    state["generated_questions"] = {}
    state["question_generation_metadata"] = {}

    for dataset_name in state["cleaned_tables"].keys():
        print(f"\n❓ Generating questions for cleaned dataset: {dataset_name}")
        
        enhanced_profile = state["data_profiles"][dataset_name]
        understanding = state["data_understanding"].get(dataset_name, {})
        
        # Verify we have clean data
        df = state["cleaned_tables"][dataset_name]
        print(f"   Clean data shape: {df.shape}")
        print(f"   Available columns: {list(df.columns)}")

        questions = agent.generate(enhanced_profile, understanding, num_questions)
        state["generated_questions"][dataset_name] = questions
        
        # Store metadata about the generation
        state["question_generation_metadata"][dataset_name] = {
            "source": "cleaned_data",
            "num_questions": len(questions),
            "dataset_shape": df.shape,
            "cleaning_impact": enhanced_profile.get("cleaning_impact", {}),
            "generation_timestamp": pd.Timestamp.now().isoformat()
        }

        print(f"\n❓ Generated {len(questions)} Questions for '{dataset_name}':")
        print("-" * 50)
        for i, q in enumerate(questions, 1):
            print(f"{i:2d}. {q}")
        print("-" * 50)

    return state