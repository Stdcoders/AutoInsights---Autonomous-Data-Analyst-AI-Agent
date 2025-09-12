import os
import pandas as pd
import plotly.express as px
import itertools
import json
import re
from typing import List, Dict, Any
#from neo_llm import NeMoLLM
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from hf_client import get_hf_client

# ---------------- Plot helpers ----------------
def plot_histogram(df: pd.DataFrame, column: str) -> str:
    fig = px.histogram(df, x=column, nbins=30, title=f"Histogram of {column}")
    return fig.to_html(full_html=False)

def plot_scatter(df: pd.DataFrame, x: str, y: str) -> str:
    fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"Scatter Plot: {x} vs {y}")
    return fig.to_html(full_html=False)

def plot_bar(df: pd.DataFrame, x: str, y: str) -> str:
    fig = px.bar(df, x=x, y=y, title=f"Bar Chart of {y} by {x}")
    return fig.to_html(full_html=False)

def plot_correlation_matrix(df: pd.DataFrame) -> str:
    corr = df.select_dtypes(include=["number"]).corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    return fig.to_html(full_html=False)

def plot_wordcloud(docs: List[str]) -> str:
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

def plot_word_frequencies(docs: List[str], top_n: int = 20) -> str:
    from collections import Counter
    all_words = " ".join(docs).split()
    counts = Counter(all_words).most_common(top_n)
    words, freqs = zip(*counts) if counts else ([], [])
    fig = px.bar(x=words, y=freqs, title="Top Word Frequencies")
    return fig.to_html(full_html=False)

def sentiment_distribution(docs: List[str], analyzer=None) -> str:
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(doc)["compound"] for doc in docs]
    fig = px.histogram(scores, nbins=20, title="Sentiment Distribution", labels={"value": "Sentiment Score"})
    return fig.to_html(full_html=False)

def text_basic_stats(docs: List[str]) -> Dict[str, Any]:
    word_counts = [len(doc.split()) for doc in docs]
    vocab = set(" ".join(docs).split())
    return {
        "num_docs": len(docs),
        "avg_length": sum(word_counts) / len(word_counts) if docs else 0,
        "vocab_size": len(vocab),
    }

# ---------------- Proactive Insight Agent ----------------
class ProactiveInsightAgent:
    def __init__(self):
        self.client, self.model_name = get_hf_client()

    def _call_hf_model(self, prompt: str) -> str:
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

    def score_insight_generic(self, answer: str, viz_html: str = None) -> float:
        score = 0.0
        text = (answer or "").lower()
        if "correl" in text: score += 3.0
        if "trend" in text or "r^2" in text or "r2" in text: score += 2.0
        if viz_html: score += 1.0
        return score

    def generate_proactive_insights(self, df: pd.DataFrame, text_columns: List[str] = None) -> List[Dict[str, Any]]:
        insights = []

        numerics = df.select_dtypes(include=["number"]).columns.tolist()
        categoricals = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Numeric pair correlations
        for col1, col2 in itertools.combinations(numerics, 2):
            corr_value = df[[col1, col2]].corr().iloc[0, 1]
            answer = f"Correlation between {col1} and {col2}: r={corr_value:.2f}"
            viz = plot_scatter(df, col1, col2)
            insights.append({"answer": answer, "visualization_html": viz, "score": self.score_insight_generic(answer, viz)})

        # Numeric distributions
        for col in numerics:
            stats_desc = df[col].describe().to_dict()
            answer = f"{col} — mean: {stats_desc['mean']:.2f}, std: {stats_desc['std']:.2f}, min: {stats_desc['min']}, max: {stats_desc['max']}"
            viz = plot_histogram(df, col)
            insights.append({"answer": answer, "visualization_html": viz, "score": self.score_insight_generic(answer, viz)})

        # Categorical top values
        for col in categoricals:
            top = df[col].value_counts(dropna=True).head(5).to_dict()
            answer = f"Top values for {col}: {top}"
            viz = plot_bar(df, x=col, y=df.index.name or "count")
            insights.append({"answer": answer, "visualization_html": viz, "score": self.score_insight_generic(answer, viz)})

        # Correlation matrix
        if len(numerics) >= 2:
            answer = "Correlation matrix for numeric columns"
            viz = plot_correlation_matrix(df)
            insights.append({"answer": answer, "visualization_html": viz, "score": self.score_insight_generic(answer, viz)})

        # Text analysis (optional) using HF for deeper insights
        if text_columns:
            for col in text_columns:
                docs = df[col].dropna().astype(str).tolist()
                stats = text_basic_stats(docs)
                
                # Use Hugging Face model for text insights
                prompt = (
                    f"You are a data analyst. Summarize the main insights of the following text data column '{col}':\n"
                    f"Sample docs: {docs[:5]}\n"
                    "Return a concise summary."
                )
                hf_answer = self._call_hf_model(prompt) or f"Text column '{col}' — {stats['num_docs']} docs, avg length {stats['avg_length']:.2f} words, vocab size {stats['vocab_size']}"

                viz = plot_wordcloud(docs)
                insights.append({"answer": hf_answer, "visualization_html": viz, "score": self.score_insight_generic(hf_answer, viz)})

        # Sort by score
        insights = sorted(insights, key=lambda x: x.get("score", 0), reverse=True)
        return insights
    def run(self, df: pd.DataFrame, text_columns: List[str] = None) -> List[Dict[str, any]]:
        """
        Entry point for LangGraph workflow.
        Takes a cleaned DataFrame and optional text columns.
        Returns a list of proactive insights with scores and visualizations.
        """
        if df is None or df.empty:
            print("⚠️ Empty DataFrame provided to ProactiveInsightAgent.run()")
            return []
        insights = self.generate_proactive_insights(df, text_columns=text_columns)
        return insights
