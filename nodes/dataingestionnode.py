import os
import pandas as pd
import json
import numpy as np
from typing import Optional, Dict, List
from collections import Counter

# Optional langdetect import - fallback if not available
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    def detect(text):
        """Fallback language detection - assumes English"""
        return 'en'

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional LangChain imports - fallback if not available
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = False
    HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    FAISS = None

try:
    from langchain_community.document_loaders import (
        CSVLoader, UnstructuredExcelLoader, JSONLoader,
        TextLoader, UnstructuredPDFLoader, UnstructuredImageLoader
    )
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_LOADERS_AVAILABLE = True
except ImportError:
    LANGCHAIN_LOADERS_AVAILABLE = False
    # Fallback classes - will handle files without LangChain
    CSVLoader = None
    UnstructuredExcelLoader = None
    JSONLoader = None
    TextLoader = None
    UnstructuredPDFLoader = None
    UnstructuredImageLoader = None
    PyPDFLoader = None
from tabulate import tabulate
from utils.state import STATE

# ================== FILE INGESTOR ==================
class FileIngestor:
    def __init__(self, embedding_model=None, chunk_size=1000, chunk_overlap=200):
        self.datasets = {}
        self.vector_stores = {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if embedding_model:
            self.embedding_model = embedding_model
        elif HUGGINGFACE_EMBEDDINGS_AVAILABLE:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embedding_model = None  # Vector search will be disabled

        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except ImportError:
            # Fallback text splitter if LangChain not available
            self.text_splitter = None

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
        
        if LANGCHAIN_LOADERS_AVAILABLE and JSONLoader:
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
        else:
            # Fallback: direct JSON loading without LangChain
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            # Create simple documents without LangChain
            documents = [Document(page_content=json.dumps(data), metadata={"source": filepath})]
        
        self._store_dataset(name, df, documents, "json")
        return df

    def from_text(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Text file not found: {filepath}")
        
        if LANGCHAIN_LOADERS_AVAILABLE and TextLoader:
            # Try UTF-8 encoding first, then fallback to default
            try:
                loader = TextLoader(filepath, encoding='utf-8')
                documents = loader.load()
            except UnicodeDecodeError:
                try:
                    loader = TextLoader(filepath, encoding='cp1252')
                    documents = loader.load()
                except UnicodeDecodeError:
                    # Final fallback to error handling
                    loader = TextLoader(filepath, encoding='utf-8', errors='ignore')
                    documents = loader.load()
            if self.text_splitter:
                split_documents = self.text_splitter.split_documents(documents)
            else:
                split_documents = documents
        else:
            # Fallback: direct text loading without LangChain
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple text splitting if no LangChain
            chunk_size = 1000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            split_documents = [Document(page_content=chunk, metadata={"source": filepath}) for chunk in chunks if chunk.strip()]
        
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
        
        if not LANGCHAIN_LOADERS_AVAILABLE:
            raise ImportError("PDF processing requires langchain-community. Please install it: pip install langchain-community")
        
        try:
            loader = UnstructuredPDFLoader(filepath)
            documents = loader.load()
        except Exception:
            loader = PyPDFLoader(filepath)
            documents = loader.load()

        if self.text_splitter:
            split_documents = self.text_splitter.split_documents(documents)
        else:
            split_documents = documents
            
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
        
        if not LANGCHAIN_LOADERS_AVAILABLE:
            raise ImportError("Image processing requires langchain-community. Please install it: pip install langchain-community")
        
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
        if self.embedding_model and documents and FAISS_AVAILABLE:
            try:
                self.vector_stores[name] = FAISS.from_documents(documents, self.embedding_model)
            except Exception as e:
                print(f"Warning: Could not create vector store for {name}: {e}")
                # Continue without vector store

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

        # Structured data profiling
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
            if LANGDETECT_AVAILABLE:
                for doc in documents[:20]:
                    try:
                        if doc.page_content.strip():  # Only detect if content exists
                            languages.append(detect(doc.page_content[:200]))
                    except Exception:
                        languages.append('unknown')  # Fallback for detection failures
            else:
                languages = ['en'] * min(len(documents), 20)  # Default to English when langdetect unavailable
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


# ================== NODE (stateless wrapper) ==================
def data_ingestion_node(dataset_name: str, file_path: str, file_type: str,
                        csv_args: Optional[dict] = None, sheet_name: Optional[str] = None):
    """
    Stateful ingestion node that stores results in global STATE.
    """
    ingestor = FileIngestor()

    if file_type == "csv":
        df = ingestor.from_csv(dataset_name, file_path, csv_args)
    elif file_type == "excel":
        df = ingestor.from_excel(dataset_name, file_path, sheet_name)
    elif file_type == "json":
        df = ingestor.from_json(dataset_name, file_path)
    elif file_type == "text":
        df = ingestor.from_text(dataset_name, file_path)
    elif file_type == "pdf":
        df = ingestor.from_pdf(dataset_name, file_path)
    elif file_type == "image":
        df = ingestor.from_image(dataset_name, file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    profile = ingestor.get_profile(dataset_name)

    # STORE IN GLOBAL STATE
    STATE.datasets[dataset_name] = df
    STATE.profiles[dataset_name] = profile
    
    # Also store the vector store if you need it later
    if hasattr(ingestor, 'vector_stores') and dataset_name in ingestor.vector_stores:
        # You might want to store vector stores in STATE too
        pass

    print(f"\n✅ Dataset '{dataset_name}' ingested successfully with {len(df)} records.")
    print("\n📊 Dataset Profile Summary:")

    if "columns" in profile and isinstance(profile["columns"], list) and all(isinstance(c, dict) for c in profile["columns"]):
        df_profile = pd.DataFrame(profile["columns"])
        print(tabulate(df_profile, headers="keys", tablefmt="pretty", showindex=False))
    else:
        print(json.dumps(profile, indent=2, default=str))

    return df, profile