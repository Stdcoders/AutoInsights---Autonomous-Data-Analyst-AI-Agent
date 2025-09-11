import pandas as pd
import plotly.express as px
import json
import re
from typing import List, Dict, Any

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


# ---------------- Insight Agent ----------------
class InsightAgent:
    def __init__(self, model: str = None):
        # Try to wire an LLM if available; otherwise fall back to heuristic-only mode
        self.llm = None
        self.model_name = model
        try:
            from gemma_llm import GemmaLLM
            self.llm = GemmaLLM(model_name=model or "gemma-small")
        except Exception:
            self.llm = None

    def _heuristic_single(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        numerics = df.select_dtypes(include=["number"]).columns.tolist()
        answer_parts = []
        viz_html = None

        answer_parts.append(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns.")

        if len(numerics) >= 2:
            # quick correlations
            corr = df[numerics].corr().abs().unstack().sort_values(ascending=False)
            corr = corr[corr < 1.0]
            top = corr.drop_duplicates().head(3)
            if not top.empty:
                colpair = top.index[0]
                score = float(top.iloc[0])
                answer_parts.append(f"Top numeric correlation: {colpair[0]} vs {colpair[1]} (r={score:.2f})")
                viz_html = plot_scatter(df, colpair[0], colpair[1])
            else:
                # fallback numeric summary
                col = numerics[0]
                stats = df[col].describe().to_dict()
                answer_parts.append(f"Numeric example `{col}` — mean: {stats.get('mean')}, std: {stats.get('std')}")
                viz_html = plot_histogram(df, col)
        elif len(numerics) == 1:
            col = numerics[0]
            stats = df[col].describe().to_dict()
            answer_parts.append(f"Numeric column `{col}` — mean: {stats.get('mean')}, std: {stats.get('std')}")
            viz_html = plot_histogram(df, col)
        else:
            cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cats:
                c = cats[0]
                top = df[c].value_counts(dropna=True).head(5).to_dict()
                answer_parts.append(f"Top values for `{c}`: {top}")

        return {"answer": " ".join(answer_parts), "visualization_html": viz_html}

    def answer(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        # Defensive: require DataFrame
        if not isinstance(df, pd.DataFrame):
            return {"answer": "No dataframe available.", "visualization_html": None}

        # Use LLM if present to produce structured JSON response (safe call)
        if self.llm:
            try:
                cols = list(df.columns)
                sample = df.head(3).to_dict(orient="records")
                prompt = f"You are a data analyst. The user asked: '{question}'.\nColumns: {cols}\nSample: {sample}\nRespond with JSON: { { 'viz_type': null, 'columns': [], 'answer': '' } }"
                # Call gemma-like API
                if hasattr(self.llm, "_call"):
                    raw = self.llm._call(prompt)
                else:
                    raw = ""

                # Extract JSON
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    viz_type = parsed.get("viz_type")
                    cols = parsed.get("columns", [])
                    answer = parsed.get("answer", "")
                    viz_html = None
                    if viz_type == "histogram" and cols:
                        viz_html = plot_histogram(df, cols[0])
                    elif viz_type == "scatter" and len(cols) >= 2:
                        viz_html = plot_scatter(df, cols[0], cols[1])
                    elif viz_type == "correlation_matrix":
                        viz_html = plot_correlation_matrix(df)
                    return {"answer": answer, "visualization_html": viz_html}
                else:
                    return self._heuristic_single(df, question)

            except Exception:
                return self._heuristic_single(df, question)

        # No LLM: deterministic heuristic
        return self._heuristic_single(df, question)

    def answer_multiple(self, df: pd.DataFrame, questions: List[str]) -> List[Dict[str, Any]]:
        results = []
        for q in questions:
            try:
                results.append({"question": q, **self.answer(df, q)})
            except Exception as e:
                results.append({"question": q, "answer": f"Error: {e}", "visualization_html": None})
        return results

    def score_insight(self, df: pd.DataFrame, res: Dict[str, Any], question: str) -> float:
        # Simple heuristic scorer: favor correlations and presence of viz
        score = 0.0
        text = (res.get("answer") or "").lower()
        if "correl" in text:
            score += 3.0
        if "trend" in text or "r^2" in text or "r2" in text:
            score += 2.0
        if res.get("visualization_html"):
            score += 1.0
        return score
