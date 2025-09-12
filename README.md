AutoInsights is a powerful, autonomous AI agent designed to automate the entire data analysis and document processing workflow. From ingesting raw data in multiple formats (CSV, Excel, PDF, Word) to generating insightful visualizations and detailed reports, this agent leverages a robust local tech stack to act as your intelligent, private data scientist.

✨ Features
Multi-Format Data Ingestion: Seamlessly load and process structured and unstructured data from CSV, Excel, PDF, and Word documents using the unstructured library and ONNX runtime.

Advanced NLP Pipeline: Perform comprehensive text processing, including tokenization, named entity recognition (NER) with spaCy, language detection, and keyword extraction with NLTK.

Local & Private Stack: Run everything on your own machine. Uses local embeddings (sentence-transformers), a local vector store (FAISS), and avoids external API calls for core processing.

Intelligent Insight Generation: Uses LangChain to structure reasoning and analysis, identifying key trends, patterns, and anomalies directly in your data.

Automated Visualization: Create a variety of compelling visualizations (histograms, scatter plots, interactive charts) with Matplotlib, Seaborn, and Plotly.

Detailed Reporting: Compiles all findings, statistics, and visualizations into a cohesive, well-structured report.

🛠️ Technology Stack
This project is built with a powerful combination of open-source libraries:

Core Data Processing: pandas, numpy

Natural Language Processing (NLP): spacy, nltk, langdetect, wordcloud

File Loading & Parsing: unstructured (with onnxruntime for inference), PyPDF2, python-docx, openpyxl

AI & Embeddings Framework: langchain, sentence-transformers

Vector Search & Storage: faiss-cpu

Data Visualization: matplotlib, seaborn, plotly

Utilities: python-dotenv, requests, regex, pillow


Currently working on integrating memory in the conversational agent. 
