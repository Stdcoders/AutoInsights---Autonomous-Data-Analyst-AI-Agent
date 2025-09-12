import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from collections import Counter
from langdetect import detect

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    CSVLoader, UnstructuredExcelLoader, JSONLoader,
    TextLoader, UnstructuredPDFLoader, UnstructuredImageLoader
)
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import json
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class FileIngestor:
    def __init__(self, embedding_model=None, chunk_size=1000, chunk_overlap=200):
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.vector_stores: Dict[str, FAISS] = {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"token": HF_TOKEN} if HF_TOKEN else {}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def ingest_file(self, name: str, filepath: str, csv_args: Optional[dict] = None, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Detect file type and ingest accordingly"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath, **(csv_args or {}))
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        elif ext == ".json":
            loader = JSONLoader(file_path=filepath, jq_schema='.', text_content=False)
            documents = loader.load()
            try:
                data = [json.loads(doc.page_content) for doc in documents]
                df = pd.DataFrame(data)
            except Exception:
                df = pd.DataFrame({
                    "content": [doc.page_content for doc in documents],
                    "metadata": [doc.metadata for doc in documents]
                })
        elif ext == ".txt":
            loader = TextLoader(filepath)
            documents = loader.load()
            split_documents = self.text_splitter.split_documents(documents)
            df = pd.DataFrame({
                "content": [doc.page_content for doc in split_documents],
                "metadata": [doc.metadata for doc in split_documents],
                "length": [len(doc.page_content) for doc in split_documents]
            })
            documents = split_documents
        elif ext == ".pdf":
            try:
                loader = UnstructuredPDFLoader(filepath)
                documents = loader.load()
            except Exception:
                loader = PyPDFLoader(filepath)
                documents = loader.load()
            split_documents = self.text_splitter.split_documents(documents)
            df = pd.DataFrame({
                "content": [doc.page_content for doc in split_documents],
                "metadata": [doc.metadata for doc in split_documents],
                "length": [len(doc.page_content) for doc in split_documents]
            })
            documents = split_documents
        elif ext in [".png", ".jpg", ".jpeg"]:
            loader = UnstructuredImageLoader(filepath)
            documents = loader.load()
            df = pd.DataFrame({
                "content": [doc.page_content for doc in documents],
                "metadata": [doc.metadata for doc in documents],
                "length": [len(doc.page_content) for doc in documents]
            })
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        self.datasets[name] = df
        if 'documents' in locals():
            self._create_vector_store(name, documents)

        print(f"✅ Dataset '{name}' ingested successfully with {len(df)} records.")
        return df

    def _create_vector_store(self, name: str, documents: List[Document]):
        if self.embedding_model and documents:
            self.vector_stores[name] = FAISS.from_documents(documents, self.embedding_model)

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())

    def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.datasets:
            raise KeyError(f"No dataset named '{name}'")
        return self.datasets[name]

    def get_vector_store(self, name: str) -> Optional[FAISS]:
        return self.vector_stores.get(name, None)
    def run(self, file_name: str, file_path: str, csv_args: Optional[dict] = None, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Main entry point for LangGraph or proactive workflow.
        Ingests the file and updates internal datasets/vector stores.
        Returns the ingested DataFrame.
        """
        df = self.ingest_file(file_name, file_path, csv_args=csv_args, sheet_name=sheet_name)
        return df
