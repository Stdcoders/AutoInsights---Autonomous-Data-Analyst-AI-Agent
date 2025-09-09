from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
class ConversationalAgent:
    """
    Conversational agent that can answer questions about ingested datasets.
    Uses vector stores created during ingestion and remembers chat history.
    """

    def __init__(self, ingestor, model: str = "gemini-1.5-pro"):
        self.ingestor = ingestor
        self.llm = ChatGoogleGenerativeAI(model=model)

        # Conversation memory (keeps chat history across turns)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Store QA chains per dataset
        self.qa_chains = {}

    def get_chain(self, dataset_name: str):
        """
        Get or create a ConversationalRetrievalChain for a dataset.
        """
        if dataset_name not in self.ingestor.vector_stores:
            raise KeyError(f"No vector store found for dataset '{dataset_name}'")

        if dataset_name not in self.qa_chains:
            retriever = self.ingestor.vector_stores[dataset_name].as_retriever()
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory
            )
            self.qa_chains[dataset_name] = chain

        return self.qa_chains[dataset_name]

    def ask(self, dataset_name: str, query: str) -> str:
        """
        Ask a conversational question about the dataset.
        """
        chain = self.get_chain(dataset_name)
        response = chain({"question": query})
        return response["answer"]

