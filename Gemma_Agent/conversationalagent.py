# ---------------- conversationalagent.py ----------------
'''from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# FROM: from langchain_google_genai import ChatGoogleGenerativeAI
# TO: Import our custom GemmaLLM class
from gemma_llm import GemmaLLM  # <-- THIS IS THE TRANSFORMERS STEP

class ConversationalAgent:
    """
    Conversational agent that can answer questions about ingested datasets.
    Combines:
    1. Vector store retrieval for text/document context
    2. Maintains conversation memory across turns
    """
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

    def __init__(self, ingestor, llm_model="google/gemma-7b-it", max_history=20): # <-- Changed default model
        self.ingestor = ingestor
        # FROM: self.llm = ChatGoogleGenerativeAI(model=llm_model)
        # TO: Use our custom class that uses transformers
        self.llm = GemmaLLM(model_name=llm_model)  # <-- THIS IS WHERE TRANSFORMERS IS USED
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_history_limit=max_history
        )
        self.qa_chains = {}

    def get_chain(self, dataset_name: str):
        if dataset_name not in self.ingestor.vector_stores:
            raise KeyError(f"No vector store found for dataset '{dataset_name}'")
        if dataset_name not in self.qa_chains:
            retriever = self.ingestor.vector_stores[dataset_name].as_retriever()
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,  # <-- This llm now uses Gemma via transformers
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
        return answer'''


# ---------------- enhanced_conversational_agent.py ----------------
'''from langchain.agents import Tool
from gemma_llm import GemmaLLM
import pandas as pd
import json

class EnhancedConversationalAgent:
    def __init__(self, state, llm_model="google/gemma-7b-it"):
        self.state = state
        self.llm = GemmaLLM(model_name=llm_model)
        
    def get_tools(self, dataset_name: str):
        """Create tools that REUSE your existing node functions"""
        tools = []
        
        # 1. REUSE Profiling Tool (from your file_ingestor)
        tools.append(Tool(
            name="data_profiling",
            func=lambda _: self._get_profile(dataset_name),
            description="Get comprehensive data profile with statistics and quality metrics"
        ))
        
        # 2. REUSE Analysis Tool (from your cleaning node)
        tools.append(Tool(
            name="data_analysis",
            func=lambda query: self._analyze_data(dataset_name, query),
            description="Run statistical analysis on cleaned data using pandas operations"
        ))
        
        # 3. REUSE Domain Understanding (from your understanding node)
        tools.append(Tool(
            name="domain_insights", 
            func=lambda _: self._get_domain_insights(dataset_name),
            description="Get LLM-generated domain understanding and context about the data"
        ))
        
        # 4. REUSE Question Suggestions (from your question generation node)
        tools.append(Tool(
            name="suggested_questions",
            func=lambda _: self._get_suggested_questions(dataset_name),
            description="Get pre-generated analytical questions for this dataset"
        ))
        
        # 5. REUSE Visualization (from your visualization node - if you have one)
        tools.append(Tool(
            name="visualization",
            func=lambda query: self._create_visualization(dataset_name, query),
            description="Generate data visualizations and charts"
        ))
        
        return tools
    
    # ------------ REUSE Existing Node Functions ------------
    
    def _get_profile(self, dataset_name: str) -> str:
        """REUSE: Your existing profiling from file_ingestor"""
        try:
            # This directly uses your already-computed profiling
            profile = self.state["file_ingestor"].get_profile(dataset_name)
            return json.dumps(profile, indent=2, default=str)
        except:
            return "Profiling not available"
    
    def _analyze_data(self, dataset_name: str, query: str) -> str:
        """REUSE: Your existing analysis capabilities"""
        try:
            # Access the ALREADY CLEANED data from your cleaning node
            df = self.state["cleaned_tables"][dataset_name]
            
            # Use the SAME analysis logic from your analysis node
            if "mean" in query.lower():
                col = self._extract_column(query, df.columns)
                return f"Mean of {col}: {df[col].mean()}"
            
            elif "sum" in query.lower():
                col = self._extract_column(query, df.columns)
                return f"Sum of {col}: {df[col].sum()}"
            
            # Add more analysis patterns from your existing node
            return f"Available analysis on cleaned data. Shape: {df.shape}, Columns: {list(df.columns)}"
            
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _get_domain_insights(self, dataset_name: str) -> str:
        """REUSE: Your existing LLM understanding"""
        # Directly use the precomputed understanding
        if "data_understanding_formatted" in self.state:
            return self.state["data_understanding_formatted"].get(dataset_name, "No insights available")
        return "Domain understanding not available"
    
    def _get_suggested_questions(self, dataset_name: str) -> str:
        """REUSE: Your existing question generation"""
        # Directly use the pre-generated questions
        if "generated_questions" in self.state:
            questions = self.state["generated_questions"].get(dataset_name, [])
            return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        return "No suggested questions available"
    
    def _create_visualization(self, dataset_name: str, query: str) -> str:
        """REUSE: Your existing visualization logic"""
        try:
            # If you have a visualization node, reuse its logic here
            df = self.state["cleaned_tables"][dataset_name]
            
            # This would call your existing visualization functions
            if "distribution" in query.lower():
                # Reuse your histogram generation logic
                return "Distribution chart generated"
            elif "correlation" in query.lower():
                # Reuse your correlation matrix logic
                return "Correlation heatmap generated"
                
            return "Visualization created based on your existing node logic"
        except:
            return "Visualization not available"
    
    # ------------ Helper Methods ------------
    def _extract_column(self, query: str, columns: list) -> str:
        for col in columns:
            if col.lower() in query.lower():
                return col
        return columns[0]  # fallback to first column

    def ask(self, dataset_name: str, query: str) -> str:
        """Smart router to use the appropriate existing tool"""
        # This logic decides which of your existing nodes to leverage
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['profile', 'statistic', 'missing', 'duplicate']):
            return self._get_profile(dataset_name)
        
        elif any(word in query_lower for word in ['mean', 'average', 'sum', 'count', 'max', 'min']):
            return self._analyze_data(dataset_name, query)
        
        elif any(word in query_lower for word in ['understand', 'domain', 'context', 'insight']):
            return self._get_domain_insights(dataset_name)
        
        elif any(word in query_lower for word in ['suggest', 'what can', 'possible']):
            return self._get_suggested_questions(dataset_name)
        
        elif any(word in query_lower for word in ['chart', 'graph', 'visualize', 'plot']):
            return self._create_visualization(dataset_name, query)
        
        else:
            return f"I can help with: profiling, analysis, insights, or visualizations. Try asking about statistics, trends, or data understanding." '''

# ---------------- enhanced_conversational_agent.py ----------------
'''from langchain.memory import ConversationBufferMemory
from gemma_llm import GemmaLLM
import pandas as pd
import json
import re

class EnhancedConversationalAgent:
    def __init__(self, state, llm_model="google/gemma-7b-it", max_history=10):
        self.state = state  # ✅ Now has access to ALL preprocessing results
        self.llm = GemmaLLM(model_name=llm_model)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_history_limit=max_history
        )
    
    def get_tools(self, dataset_name: str):
        """Tools that leverage ALL your preprocessing work"""
        tools = {}
        
        # 1. Data Analysis Tool (uses your cleaned data from dataanalysisnode)
        tools["analyze_data"] = {
            "func": lambda query: self._analyze_cleaned_data(dataset_name, query),
            "description": "Run statistical analysis on cleaned data using pandas operations"
        }
        
        # 2. Data Profile Tool (uses your file_ingestor profiling)
        tools["get_profile"] = {
            "func": lambda _: self._get_data_profile(dataset_name),
            "description": "Get comprehensive data profile with statistics and quality metrics"
        }
        
        # 3. Domain Insights Tool (uses your data_understanding_node results)
        tools["get_insights"] = {
            "func": lambda _: self._get_domain_insights(dataset_name),
            "description": "Get LLM-generated domain understanding and context"
        }
        
        # 4. Suggested Questions Tool (uses your question_generation_node results)
        tools["suggest_questions"] = {
            "func": lambda _: self._get_suggested_questions(dataset_name),
            "description": "Get pre-generated analytical questions for this dataset"
        }
        
        # 5. Visualization Tool (uses your visualizationinsightsnode)
        tools["create_visualization"] = {
            "func": lambda query: self._create_visualization(dataset_name, query),
            "description": "Generate data visualizations and charts"
        }
        
        return tools

    # ------------ Tool Implementations (REUSING your nodes) ------------
    
    def _analyze_cleaned_data(self, dataset_name: str, query: str) -> str:
        """REUSE: Your data_cleaning_analysis_node results"""
        try:
            # Access the ALREADY CLEANED data
            df = self.state["cleaned_tables"][dataset_name]
            
            # Use the SAME analysis logic from your analysis node
            if "mean" in query.lower():
                col = self._extract_column(query, df.columns)
                return f"Mean of {col}: {df[col].mean()}"
            
            elif "sum" in query.lower():
                col = self._extract_column(query, df.columns)
                return f"Sum of {col}: {df[col].sum()}"
            
            # Add more analysis patterns from your existing node
            return f"Available analysis on cleaned data. Shape: {df.shape}, Columns: {list(df.columns)}"
            
        except Exception as e:
            return f"Analysis error: {str(e)}"

    def _get_data_profile(self, dataset_name: str) -> str:
        """REUSE: Your file_ingestor profiling"""
        try:
            profile = self.state["file_ingestor"].get_profile(dataset_name)
            return json.dumps(profile, indent=2, default=str)
        except:
            return "Profiling not available"

    def _get_domain_insights(self, dataset_name: str) -> str:
        """REUSE: Your data_understanding_node results"""
        if "data_understanding_formatted" in self.state:
            return self.state["data_understanding_formatted"].get(dataset_name, "No insights available")
        return "Domain understanding not available"

    def _get_suggested_questions(self, dataset_name: str) -> str:
        """REUSE: Your question_generation_node results"""
        if "generated_questions" in self.state:
            questions = self.state["generated_questions"].get(dataset_name, [])
            return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        return "No suggested questions available"

    def _create_visualization(self, dataset_name: str, query: str) -> str:
        """REUSE: Your visualizationinsightsnode logic"""
        try:
            from visualizationinsightsnode import InsightAgent
            df = self.state["cleaned_tables"][dataset_name]
            agent = InsightAgent()
            result = agent.answer(df, query)
            return result.get("answer", "") + "\n" + ("Visualization generated" if result.get("visualization_html") else "")
        except:
            return "Visualization not available"

    # ------------ Helper Methods ------------
    def _extract_column(self, query: str, columns: list) -> str:
        for col in columns:
            if col.lower() in query.lower():
                return col
        return columns[0] if columns else None

    def ask(self, dataset_name: str, query: str) -> str:
        """Smart router that uses the appropriate preprocessing tool"""
        tools = self.get_tools(dataset_name)
        query_lower = query.lower()
        
        # Route to the right tool based on query content
        if any(word in query_lower for word in ['mean', 'average', 'sum', 'count', 'max', 'min', 'statistic']):
            return tools["analyze_data"]["func"](query)
        
        elif any(word in query_lower for word in ['profile', 'overview', 'summary', 'metadata']):
            return tools["get_profile"]["func"](None)
        
        elif any(word in query_lower for word in ['understand', 'domain', 'context', 'insight']):
            return tools["get_insights"]["func"](None)
        
        elif any(word in query_lower for word in ['suggest', 'what can', 'possible questions']):
            return tools["suggest_questions"]["func"](None)
        
        elif any(word in query_lower for word in ['chart', 'graph', 'visualize', 'plot']):
            return tools["create_visualization"]["func"](query)
        
        else:
            # Fallback to semantic search for conceptual questions
            return f"I can help with analysis, profiling, insights, or visualizations. Try asking about specific aspects of your data."'''

# ---------------- conversationalagent.py ----------------
'''from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from gemma_llm import GemmaLLM
import pandas as pd
import json
import re

class ConversationalAgent:
    """
    Enhanced conversational agent that leverages ALL preprocessing work.
    Provides fast, accurate answers using precomputed results only.
    """
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
    def get_chat_history_formatted(self) -> str:
        """Return chat history in a user-friendly formatted string"""
        history = self.memory.load_memory_variables({}).get("chat_history", [])
        if not history:
            return "No chat history available."
        
        formatted_history = []
        for i, msg in enumerate(history, 1):
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted_history.append(f"{i}. {role}: {msg.content}")
        
        return "\n".join(formatted_history)

    def __init__(self, state, llm_model="google/gemma-2b-it", max_history=10):
        self.state = state  # ✅ Access to ALL preprocessing results
        self.llm = GemmaLLM(model_name=llm_model)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_history_limit=max_history
        )
    def _has_required_preprocessing(self, dataset_name: str) -> bool:
        """Check if all required preprocessing steps are completed for this dataset"""
        required_keys = ['cleaned_tables', 'analysis_results', 'data_understanding', 'generated_questions']
        return all(key in self.state and dataset_name in self.state[key] for key in required_keys)
    def ask(self, dataset_name: str, query: str) -> str:
        """
        Smart router that uses precomputed results with absolute certainty.
        NO reprocessing - only accesses existing preprocessing results.
        """
        if not self._has_required_preprocessing(dataset_name):
            return "❌ Please run the complete preprocessing pipeline first (cleaning -> analysis -> understanding -> question generation)."
        # Add to conversation history
        self.memory.chat_memory.add_user_message(query)
        
        query_lower = query.lower()
        
        try:
            # 1. NUMERICAL QUESTIONS - Use precomputed statistics
            if any(word in query_lower for word in ['mean', 'average', 'sum', 'count', 'max', 'min', 'median', 'std', 'standard deviation']):
                response = self._answer_numerical_question(dataset_name, query_lower)
            
            # 2. DATA QUALITY QUESTIONS - Use precomputed profiling
            elif any(word in query_lower for word in ['missing', 'null', 'duplicate', 'quality', 'profile', 'overview']):
                response = self._answer_data_quality_question(dataset_name, query_lower)
            
            # 3. DOMAIN QUESTIONS - Use precomputed understanding
            elif any(word in query_lower for word in ['what is this', 'understand', 'domain', 'context', 'insight', 'about this data']):
                response = self._answer_domain_question(dataset_name)
            
            # 4. SUGGESTION QUESTIONS - Use precomputed questions
            elif any(word in query_lower for word in ['suggest', 'what can', 'possible questions', 'what should i ask']):
                response = self._answer_suggestion_question(dataset_name)
            
            # 5. COLUMN-SPECIFIC QUESTIONS - Use precomputed stats for specific columns
            elif any(col.lower() in query_lower for col in self._get_dataset_columns(dataset_name)):
                response = self._answer_column_specific_question(dataset_name, query_lower)
            
            # 6. FALLBACK - Semantic search for conceptual questions
            else:
                response = self._semantic_search_fallback(dataset_name, query)
                
        except Exception as e:
            response = f"⚠️ I cannot answer that question with certainty. Error: {str(e)}"
        
        # Add AI response to history
        self.memory.chat_memory.add_ai_message(response)
        return response

    # ------------ PRECOMPUTED ANSWER METHODS (NO REPROCESSING) ------------

    def _answer_numerical_question(self, dataset_name: str, query: str) -> str:
        """Answer numerical questions using PRECOMPUTED statistics"""
        if ('analysis_results' not in self.state or 
            dataset_name not in self.state['analysis_results']):
            return "❌ No analysis results available. Please run data analysis first."
        
        stats = self.state['analysis_results'][dataset_name]['descriptive_stats']
        
        # Extract column name from query
        target_col = None
        for col in stats.keys():
            if col.lower() in query:
                target_col = col
                break
        
        if not target_col:
            available_cols = list(stats.keys())
            return f"✅ Available numerical columns: {', '.join(available_cols)}. Ask about specific columns like 'What is the average [column_name]?'"
        
        col_stats = stats[target_col]
        
        if 'mean' in query or 'average' in query:
            return f"✅ The average {target_col} is {col_stats['mean']:,.2f}"
        elif 'sum' in query:
            total_rows = self.state['cleaned_tables'][dataset_name].shape[0]
            return f"✅ The total sum of {target_col} is {col_stats['mean'] * total_rows:,.2f}"
        elif 'max' in query or 'maximum' in query:
            return f"✅ The maximum {target_col} is {col_stats['max']:,.2f}"
        elif 'min' in query or 'minimum' in query:
            return f"✅ The minimum {target_col} is {col_stats['min']:,.2f}"
        elif 'median' in query:
            return f"✅ The median {target_col} is {col_stats['50%']:,.2f}"
        else:
            return f"✅ {target_col} statistics: Mean={col_stats['mean']:,.2f}, Min={col_stats['min']:,.2f}, Max={col_stats['max']:,.2f}"

    def _answer_data_quality_question(self, dataset_name: str, query: str) -> str:
        """Answer data quality questions using PRECOMPUTED profiling"""
        if ('cleaned_tables' not in self.state or 
            dataset_name not in self.state['cleaned_tables'] or
            'file_ingestor' not in self.state):
            return "❌ No cleaned data available. Please run data cleaning first."
        
        cleaned_df = self.state['cleaned_tables'][dataset_name]
        
        try:
            profile = self.state['file_ingestor'].get_profile(dataset_name)
        except:
            return "❌ Data profile not available."
        
        if 'missing' in query or 'null' in query:
            total_missing = sum(col.get('num_missing', 0) for col in profile.get('columns', []))
            total_cells = cleaned_df.shape[0] * cleaned_df.shape[1]
            return f"✅ Total missing values: {total_missing} ({total_missing/total_cells:.1%} of data)"
        
        elif 'duplicate' in query:
            duplicates = profile.get('num_duplicates', 0)
            return f"✅ Duplicate rows removed during cleaning: {duplicates}"
        
        elif 'quality' in query:
            if ('cleaning_report' in self.state and 
                dataset_name in self.state['cleaning_report']):
                quality_score = self.state['cleaning_report'][dataset_name]['cleaning_summary']['data_quality_score']
                return f"✅ Data quality score: {quality_score}/100"
            else:
                return "✅ Data quality assessment not available."
        
        else:
            return f"✅ Data shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns after cleaning"

    def _answer_domain_question(self, dataset_name: str) -> str:
        """Answer domain questions using PRECOMPUTED understanding"""
        if ("data_understanding_formatted" in self.state and 
            dataset_name in self.state["data_understanding_formatted"]):
            return f"✅ {self.state['data_understanding_formatted'][dataset_name]}"
        return "❌ Domain understanding not available. Please run data understanding first."

    def _answer_suggestion_question(self, dataset_name: str) -> str:
        """Answer suggestion questions using PRECOMPUTED questions"""
        if ("generated_questions" in self.state and 
            dataset_name in self.state["generated_questions"]):
            questions = self.state["generated_questions"][dataset_name]
            response = "✅ Suggested analytical questions:\n"
            for i, q in enumerate(questions[:5], 1):  # Show top 5
                response += f"{i}. {q}\n"
            return response + "\nAsk any of these questions for detailed analysis!"
        return "❌ No suggested questions available. Please run question generation first."

    def _answer_column_specific_question(self, dataset_name: str, query: str) -> str:
        """Answer questions about specific columns using PRECOMPUTED info"""
        if ('cleaned_tables' not in self.state or 
            dataset_name not in self.state['cleaned_tables']):
            return "❌ No cleaned data available."
        
        df = self.state['cleaned_tables'][dataset_name]
        
        # Find which column is being asked about
        target_col = None
        for col in df.columns:
            if col.lower() in query:
                target_col = col
                break
        
        if not target_col:
            return f"✅ Available columns: {', '.join(df.columns)}"
        
        # Try to get column info from precomputed profile
        try:
            profile = self.state['file_ingestor'].get_profile(dataset_name)
            col_info = next((col for col in profile.get('columns', []) if col['name'] == target_col), None)
            
            if col_info:
                response = f"✅ Column '{target_col}': "
                response += f"Type: {col_info['dtype']}, "
                response += f"Unique values: {col_info['num_unique']}, "
                response += f"Missing values: {col_info['num_missing']}"
                
                if 'top_values' in col_info:
                    top_vals = list(col_info['top_values'].items())[:3]
                    response += f", Top values: {dict(top_vals)}"
                    
                return response
        except:
            pass
        
        # Fallback to basic column info
        return f"✅ Column '{target_col}': {df[target_col].dtype}, {df[target_col].nunique()} unique values"

    def _semantic_search_fallback(self, dataset_name: str, query: str) -> str:
        """Fallback to semantic search for conceptual questions"""
        try:
            if ("file_ingestor" in self.state and 
                hasattr(self.state["file_ingestor"], 'vector_stores') and 
                dataset_name in self.state["file_ingestor"].vector_stores):
                
                retriever = self.state["file_ingestor"].vector_stores[dataset_name].as_retriever()
                docs = retriever.get_relevant_documents(query)
                
                if docs:
                    return f"🔍 Based on dataset content: {docs[0].page_content[:300]}..."
                else:
                    return "🤔 I don't have enough information to answer that question with certainty."
            else:
                return "🤔 Please ask about data analysis, statistics, or specific columns I can help with."
        except:
            return "🤔 I can help with data analysis questions. Try asking about statistics, data quality, or specific columns."

    def _get_dataset_columns(self, dataset_name: str) -> list:
        """Get columns from precomputed cleaned data"""
        if ("cleaned_tables" in self.state and 
            dataset_name in self.state["cleaned_tables"]):
            return list(self.state["cleaned_tables"][dataset_name].columns)
        return []

    def clear_memory(self):
        """Clear conversation history"""
        self.memory.clear()'''

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from gemma_llm import GemmaLLM  # your wrapper around Gemma model
from visualizationinsightsnode import InsightAgent
from dataanalysisnode import data_cleaning_analysis_node


class ConversationalAgent:
    """
    Conversational AI Agent for dataset analysis.
    Uses tools (querying, analysis, visualization) to respond to user queries
    instead of just acting like a chatbot.
    """

    def __init__(self, state, model_name=None):
        self.state = state
        '''self.llm = GemmaLLM(model_name=model_name)
        self.insight_agent = InsightAgent(model=model_name)'''

        # Default to Gemma for multi-step reasoning
        model_name = model_name or "google/flan-t5-base"
        self.llm = GemmaLLM(model_name=model_name)

        # Memory: keeps conversation context across turns
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # ---------------- TOOLS ----------------
        self.tools = [
            Tool(
                name="QueryData",
                func=self.query_data,
                description="Use this to answer user questions by checking dataset content."
            ),
            Tool(
                name="AnalyzeData",
                func=self.analyze_data,
                description="Use this to run statistical/cleaning analysis on the dataset."
            ),
            Tool(
                name="GenerateVisualization",
                func=self.generate_visualization,
                description="Use this to create visualizations (scatter, histogram, heatmap, etc.)."
            ),
        ]

        # ---------------- AGENT ----------------
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="conversational-react-description",
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
        )

    # --------- TOOL IMPLEMENTATIONS ---------
    def query_data(self, query: str) -> str:
        """Query dataset metadata/content."""
        if "file_ingestor" not in self.state:
            return "No dataset ingested yet."
        ingestor = self.state["file_ingestor"]
        ds_name = list(ingestor.datasets.keys())[0]
        df = ingestor.get_dataset(ds_name)

        # Simple strategy: return column names and row count
        return f"Dataset '{ds_name}' has {len(df)} rows and {len(df.columns)} columns. Columns: {list(df.columns)}"

    def analyze_data(self, query: str) -> str:
        """Run cleaning + statistical analysis on the dataset."""
        new_state = data_cleaning_analysis_node(self.state)
        self.state.update(new_state.to_dict())
        return "Data cleaned and statistical analysis performed. Results stored in state."

    def generate_visualization(self, query: str) -> str:
        """Generate a visualization using InsightAgent."""
        if "file_ingestor" not in self.state:
            return "No dataset ingested yet."
        ingestor = self.state["file_ingestor"]
        ds_name = list(ingestor.datasets.keys())[0]
        df = ingestor.get_dataset(ds_name)

        result = self.insight_agent.answer(df, query)
        self.state["last_viz"] = result.get("visualization_html")
        return result.get("answer", "Generated visualization.")

    # --------- AGENT INTERFACE ---------
    def ask(self, dataset_name: str, query: str) -> str:
        """Ask the agent a question. The agent decides which tools to use."""
        return self.agent.run(query)

    def clear_memory(self):
        """Reset conversation memory."""
        self.memory.clear()
