#!/usr/bin/env python3
"""
Simple Data Analysis Agent
A single-file AI agent for comprehensive data analysis.
"""

import os
import sys
import json
import datetime
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils.state import STATE
from nodes.dataingestionnode import data_ingestion_node
from nodes.dataanalysisnode import data_cleaning_analysis_node
from nodes.questiongenerationnode import question_generation_node, data_understanding
from nodes.visualizationinsightsnode import InsightAgent
from nodes.enhanced_reportgenerator import generate_enhanced_report

class SimpleDataAnalysisAgent:
    def __init__(self):
        self.current_dataset = None
        self.questions = []
        self.insights_agent = InsightAgent()
        
        # Enhanced state management
        self.analysis_history = {}  # {dataset_name: [q&a_sessions]}
        self.insights_graph = {}    # {dataset_name: {topic: [insights]}}
        self.dataset_metadata = {}  # {dataset_name: metadata}
        
        # Load previous session data if exists
        self.load_session_data()
        
        print("🤖 Enhanced AI Data Analysis Agent - Ready!")
        print("📊 Supports: CSV, Excel, JSON, PDF, TXT files")
        print("🧠 Context-aware analysis with memory")
        print("📁 Multi-dataset support")
        print("Type 'help' for commands or 'exit' to quit")
        print("🧠 Enhanced with LLM-driven analysis for custom questions!")

    def load_session_data(self):
        """Load previous session data from file"""
        try:
            session_file = "session_data.json"
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.analysis_history = data.get('analysis_history', {})
                    self.insights_graph = data.get('insights_graph', {})
                    self.dataset_metadata = data.get('dataset_metadata', {})
                    print(f"📚 Loaded session data: {len(self.dataset_metadata)} datasets in history")
        except Exception as e:
            print(f"⚠️ Could not load session data: {str(e)}")
    
    def save_session_data(self):
        """Save session data to file"""
        try:
            session_file = "session_data.json"
            data = {
                'analysis_history': self.analysis_history,
                'insights_graph': self.insights_graph,
                'dataset_metadata': self.dataset_metadata,
                'last_updated': datetime.datetime.now().isoformat()
            }
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Could not save session data: {str(e)}")

    def run(self):
        """Enhanced main agent loop with multi-dataset support"""
        self.show_status()  # Show initial status
        
        while True:
            try:
                # Check if we have any datasets available
                available_datasets = list(STATE.datasets.keys())
                
                if not available_datasets:
                    # No datasets loaded - ask for data upload
                    user_input = input("\n📁 Please provide the path to your data file (CSV, Excel, JSON, PDF, TXT): ").strip()
                    
                    if user_input.lower() in ['exit', 'quit']:
                        self.save_session_data()
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() == 'help':
                        self.show_help()
                        continue
                    
                    # Process the data
                    if self.load_dataset(user_input):
                        print(f"\n✅ Dataset loaded successfully!")
                        self.show_dataset_info(self.current_dataset)
                        
                else:
                    # We have datasets - enhanced command interface
                    prompt = f"\n🤖 [{self.current_dataset or 'No active dataset'}] Enter command: "
                    user_input = input(prompt).strip()
                    
                    if user_input.lower() in ['exit', 'quit']:
                        self.save_session_data()
                        print("👋 Goodbye!")
                        break
                    
                    # Handle enhanced commands
                    if not self.handle_command(user_input):
                        # If not a special command, treat as a question
                        if self.current_dataset:
                            self.answer_question_with_context(user_input)
                        else:
                            print("⚠️ No active dataset. Use 'switch <dataset>' or load a new dataset.")
                            
            except KeyboardInterrupt:
                self.save_session_data()
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"⚠️ Error: {str(e)}")

    def handle_command(self, user_input: str) -> bool:
        """Handle enhanced commands, return True if command was handled"""
        cmd = user_input.lower().strip()
        parts = cmd.split()
        
        if not parts:
            return False
            
        command = parts[0]
        
        # Dataset management commands
        if command == 'datasets':
            self.show_datasets()
            return True
        elif command == 'switch' and len(parts) > 1:
            self.switch_dataset(parts[1])
            return True
        elif command == 'active':
            self.show_active_dataset()
            return True
        elif command == 'status':
            self.show_status()
            return True
            
        # Analysis commands
        elif command == 'questions':
            self.show_questions()
            return True
        elif command == 'history':
            self.show_analysis_history()
            return True
        elif command == 'insights':
            self.show_insights()
            return True
        elif command == 'context' and len(parts) > 1:
            topic = ' '.join(parts[1:])
            self.show_context(topic)
            return True
            
        # Report and file commands
        elif command == 'report':
            self.generate_report()
            return True
        elif command == 'load' and len(parts) > 1:
            file_path = ' '.join(parts[1:]).strip('"\'')
            self.load_dataset(file_path)
            return True
            
        # Help
        elif command == 'help':
            self.show_help()
            return True
            
        return False  # Command not handled

    def show_help(self):
        """Show enhanced help with all commands"""
        print("\n🤖 Enhanced AI Data Analysis Agent - Commands:")
        print("\n📁 Dataset Management:")
        print("  load <file_path>       - Load a new dataset")
        print("  datasets               - List all loaded datasets")
        print("  switch <dataset_name>  - Switch to a different dataset")
        print("  active                 - Show active dataset details")
        print("  status                 - Show overall system status")
        
        print("\n🔍 Analysis & Questions:")
        print("  questions              - Show available questions for active dataset")
        print("  <question_text>        - Ask any question (with context awareness)")
        print("  <number>               - Ask numbered question (e.g., '1', '2')")
        print("  history                - Show previous Q&A history")
        print("  insights               - Show accumulated insights")
        print("  context <topic>        - Show context about a specific topic")
        
        print("\n📄 Reports & Output:")
        print("  report                 - Generate comprehensive report")
        
        print("\n📚 System:")
        print("  help                   - Show this help")
        print("  exit                   - Save and quit")
        
        print("\n💡 Examples:")
        print("  load sales_data.csv")
        print("  switch customers")
        print("  What are the top selling products?")
        print("  context customer behavior")

    def show_datasets(self):
        """Show all loaded datasets"""
        datasets = list(STATE.datasets.keys())
        
        if not datasets:
            print("\n📁 No datasets loaded")
            return
            
        print(f"\n📁 Available Datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            questions_count = len(STATE.questions.get(dataset, []))
            history_count = len(self.analysis_history.get(dataset, []))
            is_active = "← ACTIVE" if dataset == self.current_dataset else ""
            
            metadata = self.dataset_metadata.get(dataset, {})
            domain = metadata.get('domain', 'Unknown')
            load_time = metadata.get('load_time', 'Unknown')
            
            print(f"  {i}. {dataset} {is_active}")
            print(f"     Domain: {domain} | Questions: {questions_count} | History: {history_count}")
            print(f"     Loaded: {load_time}")
            print()
    
    def switch_dataset(self, dataset_name: str):
        """Switch to a different active dataset"""
        # Try exact match first
        if dataset_name in STATE.datasets:
            self.current_dataset = dataset_name
            self.questions = STATE.questions.get(dataset_name, [])
            STATE.set_active_dataset(dataset_name)
            print(f"\n✅ Switched to dataset: {dataset_name}")
            self.show_dataset_info(dataset_name)
            return
            
        # Try partial match
        matches = [name for name in STATE.datasets.keys() if dataset_name.lower() in name.lower()]
        if len(matches) == 1:
            self.current_dataset = matches[0]
            self.questions = STATE.questions.get(matches[0], [])
            STATE.set_active_dataset(matches[0])
            print(f"\n✅ Switched to dataset: {matches[0]}")
            self.show_dataset_info(matches[0])
        elif len(matches) > 1:
            print(f"\n⚠️ Multiple matches found: {', '.join(matches)}")
            print("Please be more specific.")
        else:
            print(f"\n❌ Dataset '{dataset_name}' not found")
            print("Available datasets:")
            for name in STATE.datasets.keys():
                print(f"  - {name}")
    
    def show_active_dataset(self):
        """Show detailed information about the active dataset"""
        if not self.current_dataset:
            print("\n⚠️ No active dataset")
            return
            
        self.show_dataset_info(self.current_dataset)
    
    def show_dataset_info(self, dataset_name: str):
        """Show detailed information about a specific dataset"""
        if dataset_name not in STATE.datasets:
            print(f"\n❌ Dataset '{dataset_name}' not found")
            return
            
        df = STATE.datasets[dataset_name]
        questions = STATE.questions.get(dataset_name, [])
        history = self.analysis_history.get(dataset_name, [])
        metadata = self.dataset_metadata.get(dataset_name, {})
        
        print(f"\n📊 Dataset: {dataset_name}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Domain: {metadata.get('domain', 'Unknown')}")
        print(f"Questions: {len(questions)}")
        print(f"Analysis History: {len(history)} sessions")
        print(f"File: {metadata.get('file_path', 'Unknown')}")
        print(f"Loaded: {metadata.get('load_time', 'Unknown')}")
        
        # Show recent insights if any
        insights = self.insights_graph.get(dataset_name, {})
        if insights:
            print(f"\n🧠 Key Insights:")
            for topic, topic_insights in list(insights.items())[:3]:  # Show top 3 topics
                print(f"  • {topic}: {len(topic_insights)} insights")
    
    def show_status(self):
        """Show overall system status"""
        datasets_count = len(STATE.datasets)
        total_questions = sum(len(questions) for questions in STATE.questions.values())
        total_history = sum(len(history) for history in self.analysis_history.values())
        
        print(f"\n📊 System Status:")
        print(f"Active Dataset: {self.current_dataset or 'None'}")
        print(f"Total Datasets: {datasets_count}")
        print(f"Total Questions: {total_questions}")
        print(f"Analysis Sessions: {total_history}")
        print(f"Session Data: {'Loaded' if self.dataset_metadata else 'New session'}")

    def load_dataset(self, file_path):
        """Load and process dataset"""
        try:
            # Clean the file path
            file_path = file_path.strip('"').strip("'")
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return False

            # Determine file type
            ext = os.path.splitext(file_path)[1].lower()
            file_type_map = {
                '.csv': 'csv', '.xlsx': 'excel', '.xls': 'excel', 
                '.json': 'json', '.pdf': 'pdf', '.txt': 'text'
            }
            
            if ext not in file_type_map:
                print(f"❌ Unsupported file type: {ext}")
                return False

            dataset_name = os.path.splitext(os.path.basename(file_path))[0].lower()
            file_type = file_type_map[ext]

            print(f"\n🔄 Processing {file_type.upper()} file...")
            
            # Step 1: Ingest data
            print("1️⃣ Ingesting data...")
            df, profile = data_ingestion_node(dataset_name, file_path, file_type)
            
            # Step 2: Clean data  
            print("2️⃣ Cleaning data...")
            data_cleaning_analysis_node(df, dataset_name)
            
            # Step 3: Domain understanding
            print("3️⃣ Understanding domain...")
            understanding = data_understanding(profile)
            STATE.understanding[dataset_name] = understanding
            domain = understanding.get('domain', 'General')
            print(f"📋 Detected domain: {domain}")
            
            # Step 4: Generate questions
            print("4️⃣ Generating questions...")
            question_generation_node(dataset_name, profile, num_questions=8)
            
            # Set active dataset after processing
            STATE.set_active_dataset(dataset_name)
            
            # Store dataset metadata
            self.dataset_metadata[dataset_name] = {
                'file_path': file_path,
                'file_type': file_type,
                'load_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'domain': domain,
                'shape': f"{df.shape[0]}x{df.shape[1]}"
            }
            
            # Initialize analysis tracking
            if dataset_name not in self.analysis_history:
                self.analysis_history[dataset_name] = []
            if dataset_name not in self.insights_graph:
                self.insights_graph[dataset_name] = {}
            
            # Set current dataset
            self.current_dataset = dataset_name
            self.questions = STATE.questions.get(dataset_name, [])
            
            # Save session data
            self.save_session_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            return False

    def answer_question_with_context(self, question: str):
        """Answer user question with context awareness"""
        try:
            # Check if it's a numbered question
            original_question = question
            if question.isdigit():
                idx = int(question) - 1
                if 0 <= idx < len(self.questions):
                    question = self.questions[idx]
                else:
                    print(f"❌ Question number {question} not found")
                    return
            elif question.lower().startswith('question '):
                try:
                    idx = int(question.split()[1]) - 1
                    if 0 <= idx < len(self.questions):
                        question = self.questions[idx]
                    else:
                        print(f"❌ Question number not found")
                        return
                except:
                    pass

            print(f"\n🔍 Analyzing with context: {question}")
            
            # Get relevant context
            context = self.get_relevant_context(question)
            
            # Get dataset
            df = STATE.datasets.get(self.current_dataset)
            if df is None:
                print("❌ Dataset not found")
                return

            # Generate enhanced prompt with context
            enhanced_question = self.enhance_question_with_context(question, context)
            
            # Generate insights
            result = self.insights_agent.answer(df, enhanced_question, self.current_dataset)
            
            # Store the analysis session
            self.store_analysis_session(original_question, question, result, context)
            
            # Display results with context indicators
            print(f"\n📊 Analysis Results:")
            if context:
                print(f"🧠 Context: Used {len(context)} previous insights")
            print(result.get('answer', 'No analysis available'))
            
            # Extract and store new insights
            self.extract_and_store_insights(question, result)
            
            # Save visualization if available
            if result.get('visualization_html'):
                self.save_visualization(result['visualization_html'], question)
            
            # Save session data
            self.save_session_data()
                
        except Exception as e:
            print(f"❌ Error analyzing question: {str(e)}")
    
    def get_relevant_context(self, question: str) -> List[Dict[str, Any]]:
        """Get relevant context from previous analysis sessions"""
        if self.current_dataset not in self.analysis_history:
            return []
        
        # Simple keyword matching for context retrieval
        question_words = set(question.lower().split())
        relevant_context = []
        
        for session in self.analysis_history[self.current_dataset]:
            # Check if questions have common keywords
            session_words = set(session['question'].lower().split())
            if question_words & session_words:  # Intersection
                relevant_context.append({
                    'question': session['question'],
                    'key_finding': session.get('key_finding', ''),
                    'timestamp': session['timestamp']
                })
        
        # Limit to most recent relevant context
        return relevant_context[-3:] if relevant_context else []
    
    def enhance_question_with_context(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Enhance question with relevant context"""
        if not context:
            return question
        
        context_str = "\nRelevant previous findings:\n"
        for i, ctx in enumerate(context, 1):
            context_str += f"{i}. From '{ctx['question']}': {ctx['key_finding']}\n"
        
        enhanced_question = f"{question}{context_str}\nPlease consider these previous findings when answering."
        return enhanced_question
    
    def store_analysis_session(self, original_input: str, processed_question: str, result: Dict[str, Any], context: List[Dict[str, Any]]):
        """Store analysis session for future context"""
        if self.current_dataset not in self.analysis_history:
            self.analysis_history[self.current_dataset] = []
        
        # Extract key finding from the result
        answer = result.get('answer', '')
        key_finding = self.extract_key_finding(answer)
        
        session = {
            'timestamp': datetime.datetime.now().isoformat(),
            'original_input': original_input,
            'question': processed_question,
            'answer': answer,
            'key_finding': key_finding,
            'context_used': len(context),
            'had_visualization': bool(result.get('visualization_html'))
        }
        
        self.analysis_history[self.current_dataset].append(session)
        
        # Keep only last 20 sessions to manage memory
        if len(self.analysis_history[self.current_dataset]) > 20:
            self.analysis_history[self.current_dataset] = self.analysis_history[self.current_dataset][-20:]
    
    def extract_key_finding(self, answer: str) -> str:
        """Extract key finding from analysis answer"""
        if not answer:
            return "No key finding"
        
        # Simple extraction - get first sentence or first 100 characters
        sentences = answer.split('. ')
        if sentences:
            return sentences[0][:100] + ('...' if len(sentences[0]) > 100 else '')
        
        return answer[:100] + ('...' if len(answer) > 100 else '')
    
    def extract_and_store_insights(self, question: str, result: Dict[str, Any]):
        """Extract and store insights in the insights graph"""
        if self.current_dataset not in self.insights_graph:
            self.insights_graph[self.current_dataset] = {}
        
        # Simple topic extraction based on question keywords
        topics = self.extract_topics(question)
        answer = result.get('answer', '')
        key_insight = self.extract_key_finding(answer)
        
        for topic in topics:
            if topic not in self.insights_graph[self.current_dataset]:
                self.insights_graph[self.current_dataset][topic] = []
            
            self.insights_graph[self.current_dataset][topic].append({
                'insight': key_insight,
                'question': question,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            # Keep only last 5 insights per topic
            if len(self.insights_graph[self.current_dataset][topic]) > 5:
                self.insights_graph[self.current_dataset][topic] = self.insights_graph[self.current_dataset][topic][-5:]
    
    def extract_topics(self, question: str) -> List[str]:
        """Extract topics from question for insight categorization"""
        # Simple topic extraction based on common data analysis keywords
        topic_keywords = {
            'sales': ['sales', 'revenue', 'profit', 'selling', 'sold'],
            'customer': ['customer', 'client', 'buyer', 'user'],
            'product': ['product', 'item', 'goods', 'merchandise'],
            'time': ['time', 'date', 'month', 'year', 'trend', 'seasonal'],
            'distribution': ['distribution', 'spread', 'range', 'variance'],
            'comparison': ['compare', 'versus', 'vs', 'difference', 'between'],
            'performance': ['performance', 'metric', 'kpi', 'score']
        }
        
        question_lower = question.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def show_analysis_history(self):
        """Show analysis history for current dataset"""
        if not self.current_dataset:
            print("\n⚠️ No active dataset")
            return
            
        history = self.analysis_history.get(self.current_dataset, [])
        if not history:
            print(f"\n📜 No analysis history for {self.current_dataset}")
            return
        
        print(f"\n📜 Analysis History for {self.current_dataset} ({len(history)} sessions):")
        print("-" * 60)
        
        for i, session in enumerate(history[-10:], 1):  # Show last 10
            timestamp = datetime.datetime.fromisoformat(session['timestamp']).strftime("%m-%d %H:%M")
            question = session['question'][:50] + ('...' if len(session['question']) > 50 else '')
            finding = session['key_finding'][:80] + ('...' if len(session['key_finding']) > 80 else '')
            context_info = f" (used {session['context_used']} context)" if session['context_used'] > 0 else ""
            viz_info = " 📈" if session['had_visualization'] else ""
            
            print(f"{i}. [{timestamp}]{context_info}{viz_info}")
            print(f"   Q: {question}")
            print(f"   ➡️ {finding}")
            print()
    
    def show_insights(self):
        """Show accumulated insights for current dataset"""
        if not self.current_dataset:
            print("\n⚠️ No active dataset")
            return
            
        insights = self.insights_graph.get(self.current_dataset, {})
        if not insights:
            print(f"\n🧠 No insights accumulated for {self.current_dataset}")
            return
        
        print(f"\n🧠 Key Insights for {self.current_dataset}:")
        print("=" * 50)
        
        for topic, topic_insights in insights.items():
            print(f"\n🔹 {topic.upper()} ({len(topic_insights)} insights):")
            for i, insight in enumerate(topic_insights[-3:], 1):  # Show last 3 per topic
                timestamp = datetime.datetime.fromisoformat(insight['timestamp']).strftime("%m-%d %H:%M")
                print(f"  {i}. [{timestamp}] {insight['insight']}")
                print(f"     From: {insight['question'][:60]}{'...' if len(insight['question']) > 60 else ''}")
    
    def show_context(self, topic: str):
        """Show context about a specific topic"""
        if not self.current_dataset:
            print("\n⚠️ No active dataset")
            return
        
        # Search in insights graph
        insights = self.insights_graph.get(self.current_dataset, {})
        topic_lower = topic.lower()
        
        relevant_insights = []
        for insight_topic, topic_insights in insights.items():
            if topic_lower in insight_topic.lower():
                relevant_insights.extend(topic_insights)
        
        # Search in analysis history
        history = self.analysis_history.get(self.current_dataset, [])
        relevant_history = []
        for session in history:
            if topic_lower in session['question'].lower() or topic_lower in session.get('key_finding', '').lower():
                relevant_history.append(session)
        
        if not relevant_insights and not relevant_history:
            print(f"\n🔍 No context found for topic: {topic}")
            return
        
        print(f"\n🔍 Context for '{topic}':")
        print("=" * 40)
        
        if relevant_insights:
            print(f"\n🧠 Insights ({len(relevant_insights)}):")
            for i, insight in enumerate(relevant_insights[-5:], 1):
                timestamp = datetime.datetime.fromisoformat(insight['timestamp']).strftime("%m-%d %H:%M")
                print(f"  {i}. [{timestamp}] {insight['insight']}")
        
        if relevant_history:
            print(f"\n📜 Recent Analysis ({len(relevant_history)} sessions):")
            for i, session in enumerate(relevant_history[-3:], 1):
                timestamp = datetime.datetime.fromisoformat(session['timestamp']).strftime("%m-%d %H:%M")
                print(f"  {i}. [{timestamp}] {session['question']}")
                print(f"     ➡️ {session['key_finding']}")

    def show_questions(self):
        """Show suggested questions"""
        if not self.questions:
            print("❌ No questions available")
            return
            
        print(f"\n💡 Suggested Questions for {self.current_dataset}:")
        for i, question in enumerate(self.questions, 1):
            print(f"{i}. {question}")
        print("\n💬 You can ask any of these questions or your own!")

    def answer_question(self, question):
        """Answer user question and generate insights"""
        try:
            # Check if it's a numbered question (e.g., "1", "question 1")
            if question.isdigit():
                idx = int(question) - 1
                if 0 <= idx < len(self.questions):
                    question = self.questions[idx]
                else:
                    print(f"❌ Question number {question} not found")
                    return
            elif question.lower().startswith('question '):
                try:
                    idx = int(question.split()[1]) - 1
                    if 0 <= idx < len(self.questions):
                        question = self.questions[idx]
                    else:
                        print(f"❌ Question number not found")
                        return
                except:
                    pass  # Use the original question

            print(f"\n🔍 Analyzing: {question}")
            
            # Get dataset
            df = STATE.datasets.get(self.current_dataset)
            if df is None:
                print("❌ Dataset not found")
                return

            # Generate insights
            result = self.insights_agent.answer(df, question, self.current_dataset)
            
            # Display results
            print(f"\n📊 Analysis Results:")
            print(result.get('answer', 'No analysis available'))
            
            # Save visualization if available
            if result.get('visualization_html'):
                self.save_visualization(result['visualization_html'], question)
                
        except Exception as e:
            print(f"❌ Error analyzing question: {str(e)}")

    def save_visualization(self, viz_html, question):
        """Save visualization to file and also try to save as PNG for PDF reports"""
        try:
            import datetime
            viz_dir = "visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.current_dataset}_{timestamp}"
            html_filepath = os.path.join(viz_dir, f"{base_filename}.html")
            png_filepath = os.path.join(viz_dir, f"{base_filename}.png")
            
            # Save HTML version
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Analysis Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h2>Question: {question}</h2>
    <div id="plot">
        {viz_html}
    </div>
</body>
</html>
                """)
            
            # Try to also save PNG version for PDF embedding
            png_success = self.save_visualization_as_png(viz_html, png_filepath)
            
            abs_html_path = os.path.abspath(html_filepath)
            print(f"📈 Visualization saved: {abs_html_path}")
            
            if png_success:
                abs_png_path = os.path.abspath(png_filepath)
                print(f"🖼️ PNG version saved: {abs_png_path}")
                # Store PNG path for report embedding
                if not hasattr(self, 'visualization_images'):
                    self.visualization_images = {}
                self.visualization_images[self.current_dataset] = self.visualization_images.get(self.current_dataset, [])
                self.visualization_images[self.current_dataset].append({
                    'png_path': abs_png_path,
                    'question': question,
                    'timestamp': timestamp
                })
            
            # Try to open HTML in browser
            try:
                import webbrowser
                webbrowser.open(f'file://{abs_html_path}')
                print("🌍 Opened in browser")
            except:
                print(f"📝 Open this file to see the visualization: {abs_html_path}")
                
        except Exception as e:
            print(f"⚠️ Could not save visualization: {e}")
    
    def save_visualization_as_png(self, viz_html, png_filepath):
        """Try to convert visualization HTML to PNG image"""
        try:
            # Method 1: Try using plotly's built-in image export
            import plotly.graph_objects as go
            import plotly.io as pio
            import re
            import json
            
            # Extract JSON from HTML (this is a simplified approach)
            # In a real implementation, you might want more robust parsing
            json_pattern = r'Plotly\.newPlot\([^,]+,\s*(\[.+?\])\s*,\s*(\{.+?\})'
            match = re.search(json_pattern, viz_html, re.DOTALL)
            
            if match:
                try:
                    data_str = match.group(1)
                    layout_str = match.group(2)
                    
                    # Clean up the JSON strings
                    data_str = re.sub(r'\n\s+', ' ', data_str)
                    layout_str = re.sub(r'\n\s+', ' ', layout_str)
                    
                    data = json.loads(data_str)
                    layout = json.loads(layout_str)
                    
                    # Create figure
                    fig = go.Figure(data=data, layout=layout)
                    
                    # Export as PNG (requires kaleido: pip install kaleido)
                    pio.write_image(fig, png_filepath, format="png", width=800, height=600, scale=2)
                    return True
                    
                except Exception as e:
                    print(f"🖼️ PNG conversion failed (JSON parsing): {e}")
                    return False
            else:
                print("🖼️ PNG conversion failed: Could not extract Plotly data from HTML")
                return False
                
        except ImportError as e:
            print(f"🖼️ PNG conversion failed: Missing dependencies ({e})")
            print("💡 To enable PNG export: pip install kaleido")
            return False
        except Exception as e:
            print(f"🖼️ PNG conversion failed: {e}")
            return False

    def generate_report(self):
        """Generate comprehensive report"""
        try:
            if not self.current_dataset:
                print("❌ No dataset loaded")
                return
                
            print(f"\n📑 Generating report for {self.current_dataset}...")
            
            output_dir = "reports"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.current_dataset}_report.pdf")
            
            # Pass analysis history to report generator so user questions are included!
            abs_path = generate_enhanced_report(STATE, self.current_dataset, output_path, self.analysis_history)
            print(f"✅ Enhanced report generated: {abs_path}")
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")

if __name__ == "__main__":
    agent = SimpleDataAnalysisAgent()
    agent.run()