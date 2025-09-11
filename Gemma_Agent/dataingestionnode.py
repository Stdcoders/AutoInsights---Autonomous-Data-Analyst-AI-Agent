import os
import pandas as pd
import json
import numpy as np
from typing import Optional, Dict, List
from collections import Counter
from langdetect import detect

from state import ReportState
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    CSVLoader, UnstructuredExcelLoader, JSONLoader,
    TextLoader, UnstructuredPDFLoader, UnstructuredImageLoader
)
from langchain_community.document_loaders import PyPDFLoader


# ================== FILE INGESTOR ==================
class FileIngestor:
    def __init__(self, embedding_model=None, chunk_size=1000, chunk_overlap=200):
        self.datasets = {}
        self.vector_stores = {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def from_csv(self, name: str, filepath: str, csv_args: Optional[Dict] = None) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        df = pd.read_csv(filepath, **(csv_args or {}))
        documents = [Document(page_content=df.to_csv(index=False), metadata={"source": filepath})]
        self._store_dataset(name, df, documents, "csv")
        return df

    def from_excel(self, name: str, filepath: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Excel file not found: {filepath}")
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        documents = [Document(page_content=df.to_csv(index=False), metadata={"source": filepath})]
        self._store_dataset(name, df, documents, "excel")
        return df

    def from_json(self, name: str, filepath: str, jq_schema: str = '.') -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        loader = JSONLoader(file_path=filepath, jq_schema=jq_schema, text_content=False)
        documents = loader.load()
        try:
            data = [json.loads(doc.page_content) for doc in documents]
            df = pd.DataFrame(data)
        except Exception:
            df = pd.DataFrame({
                "content": [doc.page_content for doc in documents],
                "metadata": [doc.metadata for doc in documents]
            })
        self._store_dataset(name, df, documents, "json")
        return df

    def from_text(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Text file not found: {filepath}")
        loader = TextLoader(filepath)
        documents = loader.load()
        split_documents = self.text_splitter.split_documents(documents)
        if not split_documents:
            raise ValueError(f"No text extracted from {filepath}")
        df = pd.DataFrame({
            "content": [doc.page_content for doc in split_documents],
            "metadata": [doc.metadata for doc in split_documents],
            "length": [len(doc.page_content) for doc in split_documents]
        })
        self._store_dataset(name, df, split_documents, "text")
        return df

    def from_pdf(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        try:
            loader = UnstructuredPDFLoader(filepath)
            documents = loader.load()
        except Exception:
            loader = PyPDFLoader(filepath)
            documents = loader.load()

        split_documents = self.text_splitter.split_documents(documents)
        if not split_documents:
            raise ValueError(f"No text extracted from {filepath}")
        df = pd.DataFrame({
            "content": [doc.page_content for doc in split_documents],
            "metadata": [doc.metadata for doc in split_documents],
            "page": [doc.metadata.get("page", 0) for doc in split_documents],
            "length": [len(doc.page_content) for doc in split_documents]
        })
        self._store_dataset(name, df, split_documents, "pdf")
        return df

    def from_image(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")
        loader = UnstructuredImageLoader(filepath)
        documents = loader.load()
        if not documents:
            raise ValueError(f"No text extracted from {filepath}")
        df = pd.DataFrame({
            "content": [doc.page_content for doc in documents],
            "metadata": [doc.metadata for doc in documents],
            "length": [len(doc.page_content) for doc in documents]
        })
        self._store_dataset(name, df, documents, "image")
        return df

    # ---------------- private helpers ----------------
    def _store_dataset(self, name: str, df: pd.DataFrame, documents: List[Document], data_type: str):
        self.datasets[name] = {
            "data": df,
            "documents": documents,
            "profile": self._profile(df, documents, data_type),
        }
        if documents:
            self._create_vector_store(name, documents)

    def _create_vector_store(self, name: str, documents: List[Document]):
        if self.embedding_model and documents:
            self.vector_stores[name] = FAISS.from_documents(documents, self.embedding_model)

    def _profile(self, df: pd.DataFrame, documents: List[Document], data_type: str) -> Dict:
        """Generate a comprehensive profile for the dataset (structured + unstructured)."""
        profile = {
            "data_type": data_type,
            "n_rows": len(df) if df is not None else 0,
            "n_cols": df.shape[1] if hasattr(df, "shape") else 1,
            "columns": list(df.columns) if hasattr(df, "columns") else ["content"],
            "document_count": len(documents),
            "sample": df.head(3).to_dict(orient="records") if df is not None else []
        }

        # Structured data
        if data_type in ["csv", "excel", "json"] and df is not None:
            profile["num_duplicates"] = int(df.duplicated().sum())
            columns = []
            for col in df.columns:
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "num_missing": int(df[col].isna().sum()),
                    "pct_missing": float(df[col].isna().mean()),
                    "num_unique": int(df[col].nunique(dropna=True)),
                }
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info.update({
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "std": df[col].std(),
                    })
                    std = df[col].std()
                    if std and std != 0:
                        z_scores = np.abs((df[col] - df[col].mean()) / std)
                        col_info["num_outliers"] = int((z_scores > 3).sum())
                    else:
                        col_info["num_outliers"] = 0
                elif pd.api.types.is_object_dtype(df[col]):
                    col_info["top_values"] = df[col].value_counts(dropna=True).head(5).to_dict()
                elif np.issubdtype(df[col].dtype, np.datetime64):
                    col_info.update({
                        "min_date": str(df[col].min()),
                        "max_date": str(df[col].max()),
                    })
                columns.append(col_info)
            profile["columns"] = columns

        # Text / PDF profiling
        if data_type in ["text", "pdf"] and documents:
            text_lengths = [len(doc.page_content) for doc in documents if doc.page_content]
            token_counts = [len(doc.page_content.split()) for doc in documents if doc.page_content]
            languages = []
            for doc in documents[:20]:
                try:
                    languages.append(detect(doc.page_content[:200]))
                except Exception:
                    continue
            profile.update({
                "avg_text_length": float(np.mean(text_lengths)) if text_lengths else 0,
                "min_text_length": int(min(text_lengths)) if text_lengths else 0,
                "max_text_length": int(max(text_lengths)) if text_lengths else 0,
                "avg_token_count": float(np.mean(token_counts)) if token_counts else 0,
                "detected_languages": dict(Counter(languages)),
            })

        # Image profiling
        if data_type == "image" and documents:
            image_meta = [doc.metadata for doc in documents if hasattr(doc, "metadata")]
            widths = [meta.get("width", 0) for meta in image_meta if "width" in meta]
            heights = [meta.get("height", 0) for meta in image_meta if "height" in meta]
            sizes = [meta.get("size_kb", 0) for meta in image_meta if "size_kb" in meta]
            profile.update({
                "num_images": len(documents),
                "avg_width": float(np.mean(widths)) if widths else None,
                "avg_height": float(np.mean(heights)) if heights else None,
                "avg_file_size_kb": float(np.mean(sizes)) if sizes else None,
            })

        return profile

    def get_profile(self, name: str) -> Dict:
        if name not in self.datasets:
            raise KeyError(f"No dataset named '{name}'")
        return self.datasets[name]["profile"]

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())

    def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.datasets:
            raise KeyError(f"No dataset named '{name}'")
        return self.datasets[name]["data"]

def format_profile_for_user(profile: dict) -> str:
    lines = []
    lines.append(f"Dataset contains {profile.get('n_rows', 0)} rows and {profile.get('n_cols', 0)} columns.\n")

    # Columns explanation
    lines.append("Columns summary:")
    for col in profile.get("columns", []):
        lines.append(f"- {col['name']} ({col['dtype']}), missing: {col.get('num_missing', 0)}, "
                     f"unique: {col.get('num_unique', 0)}")
        if "top_values" in col:
            top_vals = ", ".join([f"{k}: {v}" for k, v in list(col["top_values"].items())[:5]])
            lines.append(f"  Top values: {top_vals}")
        if "min" in col and "max" in col:
            lines.append(f"  Range: {col['min']} – {col['max']}")

    # Sample
    sample_rows = profile.get("sample", [])
    if sample_rows:
        lines.append("\nSample rows:")
        for row in sample_rows[:3]:
            row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
            lines.append(f"  {row_text}")

    # Duplicates
    lines.append(f"\nNumber of duplicate rows: {profile.get('num_duplicates', 0)}")

    return "\n".join(lines)

# ================== NODE ==================
def data_ingestion_node(
    state: ReportState, dataset_name: str, file_path: str, file_type: str,
    csv_args: Optional[dict] = None, sheet_name: Optional[str] = None
) -> ReportState:
    """This node ingests data into the pipeline and updates the ReportState."""

    if "file_ingestor" not in state:
        state["file_ingestor"] = FileIngestor()

    ingestor = state["file_ingestor"]

    if file_type == "csv":
        df = ingestor.from_csv(dataset_name, file_path, csv_args)
    elif file_type == "excel":
        df = ingestor.from_excel(dataset_name, file_path, sheet_name)
    elif file_type == "json":
        df = ingestor.from_json(dataset_name, file_path)
    '''elif file_type == "text":
        df = ingestor.from_text(dataset_name, file_path)
    elif file_type == "pdf":
        df = ingestor.from_pdf(dataset_name, file_path)
    elif file_type == "image":
        df = ingestor.from_image(dataset_name, file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")'''

    # Update state
    if "processed_tables" not in state:
        state["processed_tables"] = {}

    state["processed_tables"][dataset_name] = df

    print(f"\n✅ Dataset '{dataset_name}' ingested successfully with {len(df)} records.")

    # Show quick profile summary
    profile = ingestor.get_profile(dataset_name)
    #print("\n📊 Dataset Profile Summary:")
    
    #print(json.dumps(profile, indent=2, default=str))
    print("\n📊 Dataset Profile Summary:")
    print(format_profile_for_user(profile))


    return state