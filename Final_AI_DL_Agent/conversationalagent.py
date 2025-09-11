# ---------------- conversationalagent.py ----------------
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from visualizationinsightsnode import InsightAgent

class ConversationalAgent:
    """
    Conversational agent that can answer questions about ingested datasets.
    Combines:
    1. Vector store retrieval for text/document context
    2. Maintains conversation memory across turns
    """
    # Add this method to the ConversationalAgent class in conversationalagent.py
    def get_chat_history_formatted(self) -> str:
        """Return chat history in a user-friendly formatted string"""
        history = self.memory.load_memory_variables({}).get("chat_history", [])
        if not history:
            return "No chat history available."
    
        formatted_history = []
        for i, msg in enumerate(history, 1):
            role = "User" if msg.type == "human" else "Assistant"
            formatted_history.append(f"{i}. {role}: {msg.content}")
    
        return "\n".join(formatted_history)

# Also update the memory initialization to store more history
    def __init__(self, ingestor, llm_model="gemini-1.5-flash", max_history=20):
        self.ingestor = ingestor
        self.llm = ChatGoogleGenerativeAI(model=llm_model)
        self.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_history_limit=max_history  # Limit to prevent memory overload
        )
        self.qa_chains = {}

    def get_chain(self, dataset_name: str):
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
        df = self.ingestor.get_dataset(dataset_name)
        if df is None:
            return f"⚠️ Dataset '{dataset_name}' not found."

        chain = self.get_chain(dataset_name)
        result = chain.invoke({"question": query})
        answer = result.get("answer", "⚠️ No answer available.")
        return answer

