import os
import pandas as pd
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv

# Load .env file if you are using one
load_dotenv()

# Explicitly set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "api_key"

# LangChain + HuggingFace
from langchain_community.document_loaders import (
    CSVLoader, UnstructuredExcelLoader, JSONLoader,
    TextLoader, UnstructuredPDFLoader, UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Memory + Retrieval
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI


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
        loader = CSVLoader(file_path=filepath, csv_args=csv_args or {})
        documents = loader.load()
        data = [doc.page_content for doc in documents]
        df = pd.DataFrame([item.split(': ', 1) for item in data if ': ' in item], 
                          columns=['column', 'value'])
        self._store_dataset(name, df, documents, "csv")
        return df

    def from_excel(self, name: str, filepath: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Excel file not found: {filepath}")
        loader = UnstructuredExcelLoader(file_path=filepath, mode="elements", sheet_name=sheet_name)
        documents = loader.load()
        table_data = []
        for doc in documents:
            if 'table' in doc.metadata.get('category', ''):
                try:
                    rows = doc.page_content.split('\n')
                    if rows:
                        headers = rows[0].split('\t')
                        for row in rows[1:]:
                            values = row.split('\t')
                            if len(values) == len(headers):
                                table_data.append(dict(zip(headers, values)))
                except:
                    continue
        df = pd.DataFrame(table_data) if table_data else pd.DataFrame()
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
        except:
            df = pd.DataFrame({
                'content': [doc.page_content for doc in documents],
                'metadata': [doc.metadata for doc in documents]
            })
        self._store_dataset(name, df, documents, "json")
        return df

    def from_text(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Text file not found: {filepath}")
        loader = TextLoader(filepath)
        documents = loader.load()
        split_documents = self.text_splitter.split_documents(documents)
        df = pd.DataFrame({
            'content': [doc.page_content for doc in split_documents],
            'metadata': [doc.metadata for doc in split_documents],
            'length': [len(doc.page_content) for doc in split_documents]
        })
        self._store_dataset(name, df, split_documents, "text")
        return df

    def from_pdf(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        loader = UnstructuredPDFLoader(filepath)
        documents = loader.load()
        split_documents = self.text_splitter.split_documents(documents)
        df = pd.DataFrame({
            'content': [doc.page_content for doc in split_documents],
            'metadata': [doc.metadata for doc in split_documents],
            'page': [doc.metadata.get('page', 0) for doc in split_documents],
            'length': [len(doc.page_content) for doc in split_documents]
        })
        self._store_dataset(name, df, split_documents, "pdf")
        return df

    def from_image(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")
        loader = UnstructuredImageLoader(filepath)
        documents = loader.load()
        df = pd.DataFrame({
            'content': [doc.page_content for doc in documents],
            'metadata': [doc.metadata for doc in documents],
            'length': [len(doc.page_content) for doc in documents]
        })
        self._store_dataset(name, df, documents, "image")
        return df

    def _store_dataset(self, name: str, df: pd.DataFrame, documents: List[Document], data_type: str):
        self.datasets[name] = {
            'data': df,
            'documents': documents,
            'profile': self._profile(df, documents, data_type)
        }
        self._create_vector_store(name, documents)

    def _create_vector_store(self, name: str, documents: List[Document]):
        if documents and self.embedding_model:
            self.vector_stores[name] = FAISS.from_documents(documents, self.embedding_model)

    def _profile(self, df: pd.DataFrame, documents: List[Document], data_type: str) -> Dict:
        profile = {
            "data_type": data_type,
            "n_rows": len(df),
            "n_cols": df.shape[1] if hasattr(df, 'shape') else 1,
            "columns": list(df.columns) if hasattr(df, 'columns') else ['content'],
            "document_count": len(documents),
            "sample": df.head(3).to_dict(orient="records"),
            "description": df.describe().to_dict() if data_type in ["csv", "excel"] else None
        }
        if data_type in ["text", "pdf", "image"]:
            text_lengths = [len(doc.page_content) for doc in documents]
            profile.update({
                "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
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


# ================== MEMORY-AWARE AGENT ==================
class MemoryAwareAgent:
    def __init__(self, ingestor: FileIngestor, model="gemini-pro"):
        self.ingestor = ingestor
        self.llm = ChatGoogleGenerativeAI(model=model)
        self.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chains = {}

    def get_chain_for_dataset(self, dataset_name: str):
        if dataset_name not in self.ingestor.vector_stores:
            raise KeyError(f"No vector store for dataset '{dataset_name}'")
        if dataset_name not in self.qa_chains:
            retriever = self.ingestor.vector_stores[dataset_name].as_retriever()
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.chat_memory
            )
            self.qa_chains[dataset_name] = chain
        return self.qa_chains[dataset_name]

    def ask(self, dataset_name: str, query: str) -> str:
        chain = self.get_chain_for_dataset(dataset_name)
        response = chain({"question": query})
        return response["answer"]

    def get_profile_json(self, dataset_name: str) -> str:
        profile = self.ingestor.get_profile(dataset_name)
        return json.dumps(profile, indent=4)


# ================== MAIN ==================
if __name__ == "__main__":
    ingestor = FileIngestor()

    print("==== File Uploader ====")
    print("1. CSV\n2. Excel (XLSX)\n3. JSON\n4. Text File (.txt)\n5. PDF\n6. Image")
    choice = input("Enter your choice (1-6): ").strip()
    filepath = input("Enter full file path: ").strip()
    dataset_name = input("Enter a name for this dataset: ").strip()

    try:
        if choice == "1":
            df = ingestor.from_csv(dataset_name, filepath)
        elif choice == "2":
            df = ingestor.from_excel(dataset_name, filepath)
        elif choice == "3":
            df = ingestor.from_json(dataset_name, filepath)
        elif choice == "4":
            df = ingestor.from_text(dataset_name, filepath)
        elif choice == "5":
            df = ingestor.from_pdf(dataset_name, filepath)
        elif choice == "6":
            df = ingestor.from_image(dataset_name, filepath)
        else:
            print("❌ Invalid choice.")
            exit()

        # Initialize agent
        agent = MemoryAwareAgent(ingestor)

        # Show dataset profile JSON
        print("\n=== Dataset Profile (JSON) ===")
        print(agent.get_profile_json(dataset_name))

        # Interactive Q&A
        while True:
            query = input("\nAsk a question about this dataset (or type 'exit'): ").strip()
            if query.lower() == "exit":
                break
            answer = agent.ask(dataset_name, query)
            print("Agent:", answer)

    except Exception as e:
        print(f"❌ Error: {e}")
