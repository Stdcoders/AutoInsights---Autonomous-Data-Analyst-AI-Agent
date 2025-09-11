# hypothesisnode.py
import pandas as pd
import json
import re
from typing import List, Dict, Any
from scipy import stats
from visualizationinsightsnode import (
    plot_histogram, plot_scatter, plot_bar, plot_correlation_matrix
)

class HypothesisTesterLLM:
    """
    Generate hypotheses & use LLM to decide the best statistical test.
    Then run the test and return structured insights.
    """

    def __init__(self, llm=None):
        """
        llm: an object with a `.predict(prompt: str)` or `._call(prompt)` method
             (could be OpenAI, GemmaLLM, etc.)
        """
        self.llm = llm

    def generate_hypotheses(self, df: pd.DataFrame) -> List[str]:
        """Ask LLM to propose interesting hypotheses based on dataset schema."""
        cols = list(df.columns)
        sample = df.head(3).to_dict(orient="records")

        prompt = f"""
You are a data analyst. I have a dataset with columns: {cols}.
Here is a sample: {sample}.

Propose 5 diverse hypotheses/questions we should test,
covering distributions, correlations, group differences, trends, and anomalies.
Return them as a JSON list of strings.
"""
        try:
            if hasattr(self.llm, "_call"):
                raw = self.llm._call(prompt)
            elif hasattr(self.llm, "predict"):
                raw = self.llm.predict(prompt)
            else:
                return ["High-level overview of dataset"]

            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        # fallback
        return ["High-level overview of dataset"]

    def decide_test(self, hypothesis: str, df: pd.DataFrame) -> str:
        """Use LLM to decide the best statistical test for a hypothesis."""
        prompt = f"""
Hypothesis: "{hypothesis}"
Dataset columns: {list(df.columns)}

Which test/analysis is most appropriate?
Choose from: ["distribution", "correlation", "t-test", "anova", "trend", "categorical_count", "overview"].
Respond with JSON: {{ "test": "<choice>" }}
"""
        try:
            if hasattr(self.llm, "_call"):
                raw = self.llm._call(prompt)
            elif hasattr(self.llm, "predict"):
                raw = self.llm.predict(prompt)
            else:
                return "overview"

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                return parsed.get("test", "overview")
        except Exception:
            pass
        return "overview"

    def test_hypothesis(self, df: pd.DataFrame, hypothesis: str) -> Dict[str, Any]:
        """Run the chosen test on the dataset."""
        test_type = self.decide_test(hypothesis, df)
        result = {"hypothesis": hypothesis, "test_type": test_type,
                  "answer": "", "stats": {}, "viz_html": None, "score": 0}

        try:
            if test_type == "overview":
                result["answer"] = f"Dataset has {df.shape[0]} rows × {df.shape[1]} columns."
                result["score"] = 0.5

            elif test_type == "distribution":
                num = df.select_dtypes(include=["number"]).columns[0]
                stats_desc = df[num].describe().to_dict()
                result["answer"] = f"Distribution of {num}: mean={stats_desc['mean']:.2f}, std={stats_desc['std']:.2f}"
                result["viz_html"] = plot_histogram(df, num)
                result["score"] = 1.0

            elif test_type == "correlation":
                nums = df.select_dtypes(include=["number"]).columns[:2]
                if len(nums) >= 2:
                    r, p = stats.pearsonr(df[nums[0]].dropna(), df[nums[1]].dropna())
                    result["answer"] = f"Correlation between {nums[0]} and {nums[1]}: r={r:.2f}, p={p:.3f}"
                    result["stats"] = {"r": r, "p": p}
                    result["viz_html"] = plot_scatter(df, nums[0], nums[1])
                    result["score"] = abs(r) * 2

            elif test_type == "t-test":
                cats = df.select_dtypes(include=["object", "category"]).columns
                nums = df.select_dtypes(include=["number"]).columns
                if cats.any() and nums.any():
                    c = cats[0]; n = nums[0]
                    groups = df.dropna().groupby(c)[n].apply(list)
                    if len(groups) >= 2:
                        g1, g2 = groups.iloc[0], groups.iloc[1]
                        t, p = stats.ttest_ind(g1, g2, equal_var=False)
                        result["answer"] = f"T-test on {n} by {c}: t={t:.2f}, p={p:.3f}"
                        result["stats"] = {"t": t, "p": p}
                        result["score"] = 1.2 if p < 0.05 else 0.5

            elif test_type == "anova":
                cats = df.select_dtypes(include=["object", "category"]).columns
                nums = df.select_dtypes(include=["number"]).columns
                if cats.any() and nums.any():
                    c = cats[0]; n = nums[0]
                    groups = [g[1] for g in df.dropna().groupby(c)[n]]
                    if len(groups) >= 2:
                        f, p = stats.f_oneway(*groups)
                        result["answer"] = f"ANOVA on {n} by {c}: F={f:.2f}, p={p:.3f}"
                        result["stats"] = {"F": f, "p": p}
                        result["score"] = 1.5 if p < 0.05 else 0.5

            elif test_type == "trend":
                if "date" in " ".join(df.columns).lower():
                    time_col = [c for c in df.columns if "date" in c.lower()][0]
                    num = df.select_dtypes(include=["number"]).columns[0]
                    series = df[[time_col, num]].dropna().sort_values(time_col)
                    corr = series.reset_index().index.corr(series[num])
                    result["answer"] = f"Trend in {num} over {time_col}: correlation with time index={corr:.2f}"
                    result["score"] = abs(corr)
        except Exception as e:
            result["answer"] = f"Error while testing: {e}"

        return result
