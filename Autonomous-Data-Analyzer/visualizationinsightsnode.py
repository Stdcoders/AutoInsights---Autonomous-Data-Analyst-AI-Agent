import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
import re
import json
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

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


def text_basic_stats(docs: list[str]) -> dict:
    word_counts = [len(doc.split()) for doc in docs]
    vocab = set(" ".join(docs).split())
    return {
        "num_docs": len(docs),
        "avg_length": sum(word_counts) / len(word_counts) if docs else 0,
        "vocab_size": len(vocab),
    }

def plot_word_frequencies(docs: list[str], top_n: int = 20) -> str:
    from collections import Counter
    all_words = " ".join(docs).split()
    counts = Counter(all_words).most_common(top_n)
    words, freqs = zip(*counts) if counts else ([], [])
    fig = px.bar(x=words, y=freqs, title="Top Word Frequencies")
    return fig.to_html(full_html=False)

def plot_wordcloud(docs: list[str]) -> str:
    text = " ".join(docs)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" />'

def sentiment_distribution(docs: list[str], analyzer=None) -> str:
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(doc)["compound"] for doc in docs]
    fig = px.histogram(scores, nbins=20, title="Sentiment Distribution", labels={"value": "Sentiment Score"})
    return fig.to_html(full_html=False)



# ================= Insight Agent =================
class InsightAgent:
    def __init__(self, model: str = "gemini-1.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def answer(self, data, question: str) -> dict:
        """
        Handles both structured (DataFrame) and unstructured (list of text docs).
        Returns dict with 'answer' and 'visualization_html'.
        """
        if isinstance(data, pd.DataFrame):
            return self._answer_structured(data, question)
        elif isinstance(data, list) and all(isinstance(d, str) for d in data):
            return self._answer_text(data, question)
        else:
            raise TypeError("Data must be a pandas DataFrame or a list of text strings.")

    # ---- Structured handler ----
    def _answer_structured(self, df: pd.DataFrame, question: str) -> dict:
        cols = list(df.columns)
        prompt = f"""
        You are a data analyst. The user asked: "{question}".

        Dataset columns: {cols}

        Choose the best visualization:
        - histogram, scatter, bar, line, box, violin, heatmap, correlation_matrix, timeseries

        Return JSON only:
        {{
          "viz_type": "scatter",
          "columns": ["age", "income"],
          "answer": "There is a positive correlation between age and income."
        }}
        """
        response = self.model.generate_content(prompt, generation_config={"temperature": 0.3})
        response_text = response.candidates[0].content.parts[0].text

        try:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {"answer": response_text}
        except Exception:
            parsed = {"answer": response_text, "viz_type": None, "columns": []}

        viz_html = None
        viz_type = parsed.get("viz_type")
        cols = parsed.get("columns", [])

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

        return {"answer": parsed.get("answer", "No answer generated."), "visualization_html": viz_html}

    # ---- Text handler ----
    def _answer_text(self, docs: list[str], question: str) -> dict:
        stats = text_basic_stats(docs)
        prompt = f"""
        You are a text data analyst. The user asked: "{question}".

        Corpus stats:
        - Number of documents: {stats['num_docs']}
        - Avg document length: {stats['avg_length']:.2f} words
        - Vocabulary size: {stats['vocab_size']}

        Choose the best visualization:
        - "word_frequencies" → most common words, keywords
        - "wordcloud" → visual summary
        - "sentiment" → tone, positivity/negativity
        - "basic_stats" → length, vocabulary size

        Return JSON only:
        {{
          "viz_type": "word_frequencies",
          "answer": "The most frequent words are related to finance and money."
        }}
        """
        response = self.model.generate_content(prompt, generation_config={"temperature": 0.3})
        response_text = response.candidates[0].content.parts[0].text

        try:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {"answer": response_text}
        except Exception:
            parsed = {"answer": response_text, "viz_type": None}

        viz_html = None
        viz_type = parsed.get("viz_type")

        try:
            if viz_type == "word_frequencies":
                viz_html = plot_word_frequencies(docs)
            elif viz_type == "wordcloud":
                viz_html = plot_wordcloud(docs)
            elif viz_type == "sentiment":
                viz_html = sentiment_distribution(docs, SentimentIntensityAnalyzer())
            elif viz_type == "basic_stats":
                fig = px.bar(x=list(stats.keys()), y=list(stats.values()), title="Text Corpus Statistics")
                viz_html = fig.to_html(full_html=False)
        except Exception as e:
            parsed["answer"] += f"\n\n⚠️ Visualization failed: {e}"

        return {"answer": parsed.get("answer", "No answer generated."), "visualization_html": viz_html}