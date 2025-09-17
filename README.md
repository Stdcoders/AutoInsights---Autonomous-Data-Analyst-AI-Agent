# An autonomous, conversational data analyst agent that ingests structured and unstructured documents, processes them via NLP pipelines, builds vector embeddings, provides retrieval, generates visualizations, and produces reports. Runs wholly on local or private infrastructure for data privacy.

# Features

Ingests data from multiple formats: CSV, Excel, PDF, Word, etc.

NLP preprocessing: tokenization, language detection, Named Entity Recognition, summarization, etc.

Embedding generation + vector store search (local embeddings; FAISS)

Conversational interface (chat-based queries) over data and documents

Visualization and report generation (charts, plots, dashboards)

Modular pipeline: loaders, transformers, processors, report modules

Architecture & Project Layout
.
├── agent_worfklow_final.py        # Main entry point: orchestrates user chat, pipeline runs, ingestion → embedding → search/response → report
├── gemma_llm.py                   # Wrapper module for LLM interactions (abstraction over prompt design, model loading, etc.)
├── nodes/
│   ├── loader_*.py                # Loaders for different file types (CSV, PDF, Word, Excel, etc.)
│   ├── transformer_*.py           # Text extraction, normalization, cleaning
│   ├── processor_*.py             # NLP tasks: summarization, NER, keyword extraction, etc.
├── memory/                        # Vector store, embeddings, indexing, persistence
├── reports/                       # Templates & output logic for reports / visualizations
├── utils/                         # Utilities: file IO, plotting, logging, parsing helpers
├── models/                        # Pretrained or ONNX models, if any
├── config/ or .env / config.yaml  # Configuration variables, paths, model names, etc.
└── tests/ (if present)            # Unit tests for modules

# Requirements & Dependencies
Python ≥ 3.8
pandas
numpy
spacy
nltk
langdetect
sentence-transformers
faiss-cpu
langchain
unstructured
onnxruntime
PyPDF2
python-docx
openpyxl
matplotlib
seaborn
plotly
python-dotenv
pillow

# Command for running the agentic system 
python agent_workflow_final.py 
