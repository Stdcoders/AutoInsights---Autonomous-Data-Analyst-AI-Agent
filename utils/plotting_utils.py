# utils/plotting_utils.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from textblob import TextBlob
import plotly.io as pio

def show_plot_from_html(fig_json: str):
    """Display a Plotly chart from JSON in interactive window."""
    fig = pio.from_json(fig_json)
    fig.show()

# ================= STRUCTURED DATA PLOTS =================

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


# ================= TEXT DATA PLOTS =================

def plot_word_frequencies(docs: list[str], top_n: int = 20) -> str:
    """Plot most common words from text documents."""
    text = " ".join(docs)
    words = text.split()
    freq = Counter(words).most_common(top_n)
    df = pd.DataFrame(freq, columns=["word", "count"])
    fig = px.bar(df, x="word", y="count", title=f"Top {top_n} Words")
    return fig.to_html(full_html=False)


def plot_wordcloud(docs: list[str]) -> str:
    """Generate a word cloud from text documents."""
    text = " ".join(docs)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Convert matplotlib figure to HTML image
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return f"<img src='data:image/png;base64,{img_base64}' alt='WordCloud'/>"


def sentiment_distribution(docs: list[str]) -> str:
    """Plot sentiment polarity distribution for text documents."""
    sentiments = [TextBlob(doc).sentiment.polarity for doc in docs if doc.strip()]
    df = pd.DataFrame(sentiments, columns=["polarity"])
    fig = px.histogram(df, x="polarity", nbins=20, title="Sentiment Distribution", range_x=[-1, 1])
    return fig.to_html(full_html=False)
