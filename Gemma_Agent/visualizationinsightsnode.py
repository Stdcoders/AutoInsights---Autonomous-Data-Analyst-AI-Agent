'''import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# ================= Plotly Tools =================
def plot_histogram(df: pd.DataFrame, column: str) -> str:
    fig = px.histogram(df, x=column, nbins=30, title=f"Histogram of {column}")
    return fig.to_html(full_html=False)

def plot_scatter(df: pd.DataFrame, x: str, y: str) -> str:
    fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"Scatter Plot: {x} vs {y}")
    return fig.to_html(full_html=False)

def plot_bar(df: pd.DataFrame, x: str, y: str) -> str:
    fig = px.bar(df, x=x, y=y, title=f"Bar Chart of {y} by {x}")
    return fig.to_html(full_html=False)

def plot_line(df: pd.DataFrame, x: str, y: str) -> str:
    fig = px.line(df, x=x, y=y, title=f"Line Chart of {y} over {x}")
    return fig.to_html(full_html=False)

def plot_box(df: pd.DataFrame, column: str) -> str:
    fig = px.box(df, y=column, title=f"Box Plot of {column}")
    return fig.to_html(full_html=False)

def plot_violin(df: pd.DataFrame, column: str) -> str:
    fig = px.violin(df, y=column, box=True, points="all", title=f"Violin Plot of {column}")
    return fig.to_html(full_html=False)

def plot_heatmap(df: pd.DataFrame, columns: list) -> str:
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    return fig.to_html(full_html=False)

def plot_correlation_matrix(df: pd.DataFrame) -> str:
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, title="Full Correlation Matrix")
    return fig.to_html(full_html=False)

def plot_timeseries(df: pd.DataFrame, time_col: str, value_col: str) -> str:
    fig = px.line(df, x=time_col, y=value_col, title=f"Time Series of {value_col} over {time_col}")
    return fig.to_html(full_html=False)


# ================= Insight Agent =================
class InsightAgent:
    def __init__(self, model: str = "gemini-1.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def answer(self, df: pd.DataFrame, question: str) -> dict:
        """
        Use LLM to answer a question and decide which visualization(s) to generate.
        Returns dict with answer + visualization HTML.
        """
        cols = list(df.columns)

        # Build prompt for the LLM
        prompt = f"""
        You are a data analyst. The user asked: "{question}".

        Dataset columns: {cols}

        1. Decide which visualization (histogram, scatter, bar, line, box, violin, heatmap, correlation_matrix, timeseries) 
           best answers this question.
        2. Specify the required columns.
        3. Provide a short natural language answer based on the dataset.

        Return your response strictly as JSON with keys:
        "viz_type", "columns", "answer".
        Example:
        {{
          "viz_type": "scatter",
          "columns": ["age", "income"],
          "answer": "There is a positive correlation between age and income."
        }}
        """

        response = self.model.generate_content(
            prompt, generation_config={"temperature": 0.3, "max_output_tokens": 500}
        )
        response_text = response.candidates[0].content.parts[0].text

        try:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {"answer": response_text}
        except Exception:
            parsed = {"answer": response_text, "viz_type": None, "columns": []}

        viz_html = None
        viz_type = parsed.get("viz_type")
        cols = parsed.get("columns", [])

        # Dispatch to correct visualization
        try:
            if viz_type == "histogram" and cols:
                viz_html = plot_histogram(df, cols[0])
            elif viz_type == "scatter" and len(cols) >= 2:
                viz_html = plot_scatter(df, cols[0], cols[1])
            elif viz_type == "bar" and len(cols) >= 2:
                viz_html = plot_bar(df, cols[0], cols[1])
            elif viz_type == "line" and len(cols) >= 2:
                viz_html = plot_line(df, cols[0], cols[1])
            elif viz_type == "box" and cols:
                viz_html = plot_box(df, cols[0])
            elif viz_type == "violin" and cols:
                viz_html = plot_violin(df, cols[0])
            elif viz_type == "heatmap" and cols:
                viz_html = plot_heatmap(df, cols)
            elif viz_type == "correlation_matrix":
                viz_html = plot_correlation_matrix(df)
            elif viz_type == "timeseries" and len(cols) >= 2:
                viz_html = plot_timeseries(df, cols[0], cols[1])
        except Exception as e:
            parsed["answer"] += f"\n\n⚠️ Visualization failed: {e}"

        return {
            "answer": parsed.get("answer", "No answer generated."),
            "visualization_html": viz_html,
        }'''

import pandas as pd
import plotly.express as px
import os
import json
import re
from gemma_llm import GemmaLLM  # Using your local Gemma model

# ================= Plotly Tools =================
def plot_histogram(df: pd.DataFrame, column: str) -> str:
    fig = px.histogram(df, x=column, nbins=30, title=f"Histogram of {column}")
    return fig.to_html(full_html=False)

def plot_scatter(df: pd.DataFrame, x: str, y: str) -> str:
    fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"Scatter Plot: {x} vs {y}")
    return fig.to_html(full_html=False)

def plot_bar(df: pd.DataFrame, x: str, y: str) -> str:
    fig = px.bar(df, x=x, y=y, title=f"Bar Chart of {y} by {x}")
    return fig.to_html(full_html=False)

def plot_line(df: pd.DataFrame, x: str, y: str) -> str:
    fig = px.line(df, x=x, y=y, title=f"Line Chart of {y} over {x}")
    return fig.to_html(full_html=False)

def plot_box(df: pd.DataFrame, column: str) -> str:
    fig = px.box(df, y=column, title=f"Box Plot of {column}")
    return fig.to_html(full_html=False)

def plot_violin(df: pd.DataFrame, column: str) -> str:
    fig = px.violin(df, y=column, box=True, points="all", title=f"Violin Plot of {column}")
    return fig.to_html(full_html=False)

def plot_heatmap(df: pd.DataFrame, columns: list) -> str:
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    return fig.to_html(full_html=False)

def plot_correlation_matrix(df: pd.DataFrame) -> str:
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, title="Full Correlation Matrix")
    return fig.to_html(full_html=False)

def plot_timeseries(df: pd.DataFrame, time_col: str, value_col: str) -> str:
    fig = px.line(df, x=time_col, y=value_col, title=f"Time Series of {value_col} over {time_col}")
    return fig.to_html(full_html=False)


# ================= Insight Agent with Gemma =================
class InsightAgent:
    def __init__(self, model: str = "google/gemma-2b-it"):
        self.llm = GemmaLLM(model_name=model)  # Using local Gemma model

    def answer(self, df: pd.DataFrame, question: str) -> dict:
        """
        Use Gemma LLM to answer a question and decide which visualization(s) to generate.
        Returns dict with answer + visualization HTML.
        """
        cols = list(df.columns)
        sample_data = df.head(3).to_dict(orient='records')

        # Build prompt for Gemma LLM
        prompt = f"""
        You are a data analyst. The user asked: "{question}".

        Dataset columns: {cols}
        Sample data (first 3 rows): {sample_data}

        Instructions:
        1. Analyze the question and determine the most appropriate visualization type
        2. Select the required columns from the available columns
        3. Provide a concise analytical answer based on the data

        Available visualization types: histogram, scatter, bar, line, box, violin, heatmap, correlation_matrix, timeseries

        Return your response as JSON with these exact keys:
        - "viz_type": the visualization type
        - "columns": list of column names needed
        - "answer": your analytical answer

        Example response:
        {{
          "viz_type": "scatter",
          "columns": ["age", "income"],
          "answer": "There appears to be a positive correlation between age and income based on the data distribution."
        }}
        """

        response_text = self.llm._call(prompt)

        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = {"answer": response_text, "viz_type": None, "columns": []}
        except Exception as e:
            parsed = {"answer": f"Analysis completed. {response_text}", "viz_type": None, "columns": []}

        viz_html = None
        viz_type = parsed.get("viz_type")
        cols = parsed.get("columns", [])

        # Dispatch to correct visualization
        try:
            if viz_type == "histogram" and cols:
                viz_html = plot_histogram(df, cols[0])
            elif viz_type == "scatter" and len(cols) >= 2:
                viz_html = plot_scatter(df, cols[0], cols[1])
            elif viz_type == "bar" and len(cols) >= 2:
                viz_html = plot_bar(df, cols[0], cols[1])
            elif viz_type == "line" and len(cols) >= 2:
                viz_html = plot_line(df, cols[0], cols[1])
            elif viz_type == "box" and cols:
                viz_html = plot_box(df, cols[0])
            elif viz_type == "violin" and cols:
                viz_html = plot_violin(df, cols[0])
            elif viz_type == "heatmap" and cols:
                viz_html = plot_heatmap(df, cols)
            elif viz_type == "correlation_matrix":
                viz_html = plot_correlation_matrix(df)
            elif viz_type == "timeseries" and len(cols) >= 2:
                viz_html = plot_timeseries(df, cols[0], cols[1])
        except Exception as e:
            parsed["answer"] += f"\n\n⚠️ Visualization failed: {str(e)}"

        return {
            "answer": parsed.get("answer", "Analysis completed successfully."),
            "visualization_html": viz_html,
        }

    def simple_visualization(self, df: pd.DataFrame, viz_type: str, columns: list) -> dict:
        """
        Direct visualization without LLM analysis - for predictable cases
        """
        viz_html = None
        answer = f"Generated {viz_type} visualization for columns: {columns}"
        
        try:
            if viz_type == "histogram" and columns:
                viz_html = plot_histogram(df, columns[0])
                answer = f"Histogram of {columns[0]} showing data distribution"
            elif viz_type == "scatter" and len(columns) >= 2:
                viz_html = plot_scatter(df, columns[0], columns[1])
                answer = f"Scatter plot of {columns[0]} vs {columns[1]} showing relationship"
            elif viz_type == "bar" and len(columns) >= 2:
                viz_html = plot_bar(df, columns[0], columns[1])
                answer = f"Bar chart of {columns[1]} by {columns[0]}"
            elif viz_type == "correlation_matrix":
                viz_html = plot_correlation_matrix(df)
                answer = "Correlation matrix showing relationships between all numerical columns"
        except Exception as e:
            answer = f"Visualization failed: {str(e)}"
        
        return {
            "answer": answer,
            "visualization_html": viz_html
        }
