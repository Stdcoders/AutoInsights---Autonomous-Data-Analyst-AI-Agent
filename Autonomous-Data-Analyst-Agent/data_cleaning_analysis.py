from state import ReportState
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
#nltk.download("punkt")
#nltk.download("stopwords")
#nltk.download("wordnet")


class DataCleaningNode:
    def __init__(self, state: ReportState):
        self.state = state

    def run(self):
        """
        Main entry point for LangGraph or proactive workflow.
        Cleans all processed tables and updates the state.
        """
        if "processed_tables" not in self.state:
            raise ValueError("No processed tables found in state.")

        processed_tables = self.state["processed_tables"]

        self.state["cleaned_tables"] = {}
        self.state["analysis_results"] = {}
        self.state["cleaning_report"] = {}

        for dataset_name, df in processed_tables.items():
            print(f"\n🧹 Cleaning dataset: {dataset_name}")

            if "content" in df.columns and df.shape[1] <= 3:
                # ================= Text Dataset =================
                self._clean_text_dataset(dataset_name, df)
            else:
                # ================= Structured Dataset =================
                self._clean_structured_dataset(dataset_name, df)

        return self.state

    # ----------------- Internal Methods -----------------

    def _clean_text_dataset(self, dataset_name, df):
        cleaning_report = {
            "original_shape": df.shape,
            "text_cleaned": True,
            "cleaning_steps": []
        }

        cleaned_df = df.copy()
        cleaned_df["cleaned_content"] = cleaned_df["content"].apply(clean_text_document)

        cleaning_report["final_shape"] = cleaned_df.shape
        cleaning_report["cleaning_steps"].append(
            "Normalized, tokenized, stopwords removed, lemmatized"
        )

        # Text analysis
        word_counts = cleaned_df["cleaned_content"].str.split().apply(len)
        analysis_results = {
            "avg_doc_length": word_counts.mean(),
            "min_doc_length": word_counts.min(),
            "max_doc_length": word_counts.max(),
            "top_words": pd.Series(
                " ".join(cleaned_df["cleaned_content"]).split()
            ).value_counts().head(20).to_dict()
        }

        self.state["cleaned_tables"][dataset_name] = cleaned_df
        self.state["analysis_results"][dataset_name] = analysis_results
        self.state["cleaning_report"][dataset_name] = cleaning_report

        print(f"✅ Text dataset '{dataset_name}' cleaned with {len(cleaned_df)} records.")
        print(f"Average document length: {analysis_results['avg_doc_length']:.2f}")

    def _clean_structured_dataset(self, dataset_name, df):
        cleaning_report = {
            "original_shape": df.shape,
            "missing_values_before": df.isnull().sum().to_dict(),
            "duplicates_removed": 0,
            "outliers_handled": {},
            "columns_cleaned": {},
            "data_type_changes": {}
        }

        cleaned_df = df.copy()

        cleaned_df = handle_missing_values(cleaned_df, cleaning_report)
        duplicates_before = cleaned_df.duplicated().sum()
        cleaned_df = cleaned_df.drop_duplicates()
        cleaning_report["duplicates_removed"] = duplicates_before

        cleaned_df = standardize_data_types(cleaned_df, cleaning_report)
        cleaned_df = clean_string_columns(cleaned_df, cleaning_report)
        cleaned_df = handle_outliers(cleaned_df, cleaning_report)
        cleaned_df = validate_data_ranges(cleaned_df, cleaning_report)
        cleaned_df = create_new_features(cleaned_df, cleaning_report)

        cleaning_report["final_shape"] = cleaned_df.shape
        cleaning_report["missing_values_after"] = cleaned_df.isnull().sum().to_dict()
        cleaning_report["cleaning_summary"] = generate_cleaning_summary(cleaned_df, df)

        analysis_results = perform_comprehensive_analysis(cleaned_df)

        self.state["cleaned_tables"][dataset_name] = cleaned_df
        self.state["analysis_results"][dataset_name] = analysis_results
        self.state["cleaning_report"][dataset_name] = cleaning_report

        print(f"✅ Structured dataset '{dataset_name}' cleaned and analyzed.")
        print(f"Final shape: {cleaned_df.shape}, Removed {duplicates_before} duplicates.")

# ----------------- Helper Functions -----------------

def clean_text_document(text: str) -> str:
    """Clean a single text string (used for TXT/PDF content)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def handle_missing_values(df, report):
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            missing_count = df[column].isnull().sum()
            if df[column].dtype in ["int64", "float64"]:
                fill = df[column].median()
            elif df[column].dtype == "object":
                fill = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
            else:
                fill = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
            df[column] = df[column].fillna(fill)
            report["columns_cleaned"][column] = f"Filled {missing_count} missing values with {fill}"
    return df


def standardize_data_types(df, report):
    for column in df.columns:
        original_dtype = str(df[column].dtype)
        if any(k in column.lower() for k in ["date", "time", "timestamp"]):
            try:
                df[column] = pd.to_datetime(df[column], errors="coerce")
                report["data_type_changes"][column] = f"{original_dtype} -> datetime"
            except:
                pass
        elif df[column].dtype == "object" and df[column].str.isnumeric().all():
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                report["data_type_changes"][column] = f"{original_dtype} -> numeric"
            except:
                pass
    return df


def clean_string_columns(df, report):
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.strip()
        if any(k in column.lower() for k in ["name", "title", "description"]):
            df[column] = df[column].str.title()
        elif any(k in column.lower() for k in ["code", "id", "abbreviation"]):
            df[column] = df[column].str.upper()
        df[column] = df[column].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)) if pd.notnull(x) else x)
        report["columns_cleaned"][column] = "Standardized text formatting"
    return df


def handle_outliers(df, report, method="iqr"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower) | (df[column] > upper)]
            if not outliers.empty:
                df[column] = np.where(df[column] < lower, lower, df[column])
                df[column] = np.where(df[column] > upper, upper, df[column])
                report["outliers_handled"][column] = f"Capped {len(outliers)} outliers"
    return df


def validate_data_ranges(df, report):
    validation_rules = {
        "age": (0, 120),
        "salary": (0, 1000000),
        "rating": (1, 5),
    }
    for column, (min_val, max_val) in validation_rules.items():
        if column in df.columns:
            invalid = df[(df[column] < min_val) | (df[column] > max_val)]
            if not invalid.empty:
                df[column] = df[column].clip(min_val, max_val)
                report["columns_cleaned"][column] = f"Clipped {len(invalid)} values to [{min_val}, {max_val}]"
    return df


def create_new_features(df, report):
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for col in date_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            new_col = f"{col}_year"
            df[new_col] = df[col].dt.year
            report["columns_cleaned"][new_col] = "Created new feature from date"
    return df


def generate_cleaning_summary(cleaned_df, original_df):
    summary = {
        "rows_removed": original_df.shape[0] - cleaned_df.shape[0],
        "columns_remaining": cleaned_df.shape[1],
        "total_missing_values_removed": original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum(),
        "data_quality_score": calculate_data_quality_score(cleaned_df)
    }
    return summary


def calculate_data_quality_score(df):
    completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    uniqueness = 1 - (df.duplicated().sum() / df.shape[0])
    return round((completeness * 0.7 + uniqueness * 0.3) * 100, 2)


def perform_comprehensive_analysis(df):
    analysis = {
        "descriptive_stats": df.describe().to_dict(),
        "correlation_matrix": df.select_dtypes(include=[np.number]).corr().to_dict(),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "unique_values": {col: df[col].nunique() for col in df.columns},
        "missing_values": df.isnull().sum().to_dict()
    }
    return analysis
