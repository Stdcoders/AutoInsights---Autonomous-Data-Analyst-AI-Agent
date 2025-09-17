import os
import json
import re
import pandas as pd
import numpy as np
from tabulate import tabulate

from gemma_llm import GemmaLLM
from utils.state import STATE
# ===================== Data Understanding =====================

def data_understanding(df_profile: dict, model: str = "gemini-1.5-flash-latest") -> dict:  # FIXED MODEL NAME
    """
    Use HuggingFaceHub LLM to semantically understand the dataset domain and context.
    For text documents (PDFs, text files), analyzes actual content.
    For structured data, analyzes column names and characteristics.
    Returns structured understanding (domain, column_roles, use_cases, limitations).
    """
    data_type = df_profile.get('data_type', 'unknown')
    
    # Handle text documents differently (PDFs, text files)
    if data_type in ['pdf', 'text']:
        return _text_document_understanding(df_profile)
    else:
        return _structured_data_understanding(df_profile)


def _text_document_understanding(df_profile: dict) -> dict:
    """
    Analyze text documents (PDFs, text files) based on actual content.
    """
    try:
        # Initialize Gemma LLM
        llm = GemmaLLM(
            temperature=0.3,
            max_tokens=300
        )

        # Get sample content from the document
        sample_content = ""
        if 'sample' in df_profile and df_profile['sample']:
            # Extract text content from samples
            for sample in df_profile['sample'][:3]:  # Use first 3 samples
                if isinstance(sample, dict) and 'content' in sample:
                    sample_content += sample['content'][:500] + " "  # First 500 chars per sample
        
        # Truncate to manageable size for LLM
        sample_content = sample_content[:1500]  # Keep first 1500 characters
        
        prompt = f"""
        You are analyzing a text document. Based on the content below, determine the domain and purpose.

        Document info:
        - Type: {df_profile.get('data_type', 'text document')}
        - Pages/Chunks: {df_profile.get('n_rows', 'Unknown')}
        - Language: {df_profile.get('detected_languages', {})}
        
        Sample content:
        {sample_content}
        
        Based on this content, determine:
        1. Domain: What field/topic does this document cover? (technology, business, healthcare, education, research, etc.)
        2. Document type: What kind of document is this? (research paper, report, manual, etc.)
        3. Main topics: What are the key topics discussed?
        
        Be specific and base your analysis on the actual content.
        """

        if llm.is_available():  # Check if LLM is available
            response_text = llm(prompt)
            
            # Try to parse structured response, fallback to raw text
            domain = _extract_domain_from_content(sample_content, response_text)
            
            understanding = {
                "domain": domain,
                "analysis": response_text,
                "document_type": df_profile.get('data_type', 'text'),
                "content_length": len(sample_content),
                "chunks_count": df_profile.get('n_rows', 0)
            }
        else:
            return _fallback_text_understanding(df_profile, sample_content)

    except Exception as e:
        print(f"⚠️ LLM analysis failed: {e}. Using fallback analysis.")
        understanding = _fallback_text_understanding(df_profile, sample_content)

    # Pretty Console Output
    print(f"\n🤖 Data Understanding:")
    print(f"🌐 Domain: {understanding.get('domain', 'General')}")
    print(f"📊 Analysis: {understanding.get('analysis', 'Basic profile analysis')}")
    
    return understanding


def _structured_data_understanding(df_profile: dict) -> dict:
    """
    Analyze structured data (CSV, Excel, JSON) based on column names and characteristics.
    """
    try:
        # Initialize Gemma LLM
        llm = GemmaLLM(
            temperature=0.3,
            max_tokens=300
        )

        column_names = [col.get('name', 'Unknown') if isinstance(col, dict) else str(col) for col in df_profile.get('columns', [])[:10]]
        
        prompt = f"""
        You are a data expert analyzing any type of dataset. Do not assume a specific domain.

        Dataset profile summary:
        - Rows: {df_profile.get('n_rows', 'Unknown')}
        - Columns: {df_profile.get('n_cols', 'Unknown')}
        - Data type: {df_profile.get('data_type', 'Unknown')}
        - Column names: {column_names}

        Based ONLY on the column names and data characteristics, determine:
        1. Domain: What type of data is this? (can be ANY domain - sports, education, weather, social media, manufacturing, etc.)
        2. Main purpose: What might this data be used for?
        3. Key insights: What patterns could we find?

        Be data-driven and don't assume any default domain. If unclear, say "Mixed/General".
        """

        if llm.is_available():  # Check if LLM is available
            response_text = llm(prompt)
            
            # Try to parse structured response, fallback to raw text
            domain = "General"
            # Extract domain from response
            import re
            domain_match = re.search(r'Domain[:\s]*([^\n,]+)', response_text, re.IGNORECASE)
            if domain_match:
                domain = domain_match.group(1).strip()
            
            understanding = {
                "domain": domain,
                "analysis": response_text,
                "column_count": df_profile.get('n_cols', 0),
                "row_count": df_profile.get('n_rows', 0)
            }
        else:
            return _fallback_understanding(df_profile)

    except Exception as e:
        print(f"⚠️ LLM analysis failed: {e}. Using fallback analysis.")
        understanding = _fallback_understanding(df_profile)

    return understanding


def _extract_domain_from_content(content: str, llm_response: str) -> str:
    """
    Extract domain from document content and LLM response.
    Uses comprehensive keyword analysis to detect ANY domain.
    """
    content_lower = content.lower()
    
    # First try to extract from LLM response
    import re
    domain_match = re.search(r'Domain[:\s]*([^\n,]+)', llm_response, re.IGNORECASE)
    if domain_match:
        return domain_match.group(1).strip()
    
    # Expanded domain detection for any type of content
    domain_keywords = {
        "Technology/AI/ML": ['machine learning', 'automl', 'predictive analytics', 'algorithm', 'neural', 
                             'artificial intelligence', 'deep learning', 'data science', 'programming', 
                             'software', 'computer', 'technology', 'automation', 'digital'],
        "Academic/Research": ['research', 'study', 'analysis', 'methodology', 'findings', 'conclusion', 
                              'hypothesis', 'experiment', 'academic', 'scholar', 'publication', 
                              'literature review', 'theoretical', 'empirical'],
        "Business/Finance": ['business', 'market', 'strategy', 'revenue', 'customer', 'sales', 
                             'profit', 'finance', 'investment', 'economy', 'commercial', 'corporate', 
                             'management', 'leadership', 'marketing'],
        "Healthcare/Medical": ['health', 'medical', 'patient', 'treatment', 'clinical', 'diagnosis', 
                               'therapy', 'medicine', 'hospital', 'doctor', 'nurse', 'disease', 
                               'symptoms', 'healthcare', 'pharmaceutical'],
        "Education/Learning": ['education', 'learning', 'student', 'course', 'university', 'academic', 
                               'teaching', 'curriculum', 'school', 'training', 'pedagogy', 
                               'educational', 'instructor'],
        "Legal/Law": ['legal', 'law', 'court', 'judge', 'attorney', 'regulation', 'compliance', 
                      'legislation', 'contract', 'litigation', 'judicial', 'statute'],
        "Science/Engineering": ['science', 'engineering', 'physics', 'chemistry', 'biology', 'mathematics', 
                                'scientific', 'technical', 'laboratory', 'experiment', 'innovation'],
        "Arts/Culture": ['art', 'culture', 'creative', 'design', 'music', 'literature', 'history', 
                         'cultural', 'artistic', 'aesthetic', 'humanities'],
        "Sports/Recreation": ['sports', 'game', 'player', 'team', 'competition', 'athletic', 'fitness', 
                              'recreation', 'exercise', 'training', 'performance'],
        "Travel/Tourism": ['travel', 'tourism', 'destination', 'hotel', 'vacation', 'journey', 
                           'tourist', 'hospitality', 'adventure', 'exploration'],
        "Food/Nutrition": ['food', 'nutrition', 'recipe', 'cooking', 'diet', 'meal', 'restaurant', 
                           'culinary', 'ingredient', 'health food'],
        "Environment/Sustainability": ['environment', 'sustainability', 'climate', 'ecology', 'green', 
                                       'renewable', 'conservation', 'pollution', 'carbon', 'ecosystem'],
        "Government/Policy": ['government', 'policy', 'politics', 'public', 'administration', 'governance', 
                              'political', 'municipal', 'federal', 'regulation']
    }
    
    # Count matches for each domain
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            domain_scores[domain] = score
    
    # Return the domain with highest score, or "General" if no matches
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        # Only return specific domain if we have reasonable confidence (at least 2 keyword matches)
        if domain_scores[best_domain] >= 2:
            return best_domain
    
    return "General"


def _fallback_text_understanding(df_profile: dict, content: str) -> dict:
    """
    Fallback text analysis when LLM is not available.
    """
    domain = _extract_domain_from_content(content, "")
    
    return {
        "domain": domain,
        "analysis": f"Text document with {df_profile.get('n_rows', 0)} chunks. Content analysis suggests {domain} domain based on keywords.",
        "document_type": df_profile.get('data_type', 'text'),
        "content_length": len(content),
        "chunks_count": df_profile.get('n_rows', 0),
        "fallback": True
    }


def _fallback_understanding(df_profile: dict) -> dict:
    """Fallback analysis when LLM is not available."""
    columns = df_profile.get('columns', [])
    data_type = df_profile.get('data_type', 'unknown')
    
    # Flexible heuristics for domain detection based on actual data
    column_names = []
    if isinstance(columns, list):
        for col in columns:
            if isinstance(col, dict):
                column_names.append(col.get('name', '').lower())
            else:
                column_names.append(str(col).lower())
    
    # Expanded domain detection - can handle ANY type of data
    domain = "Mixed/General"
    all_columns = ' '.join(column_names)
    
    # Sleep/Health domain (PRIORITY - check first)
    if any(word in all_columns for word in ['sleep', 'duration', 'quality', 'bedtime', 'wake', 'rest', 'dream']):
        domain = "Sleep/Health"
    # Healthcare domain (check early for health-related terms)
    elif any(word in all_columns for word in ['patient', 'diagnosis', 'treatment', 'medical', 'health', 'symptom', 'disease', 'stress level', 'bmi', 'blood pressure', 'heart rate']):
        domain = "Healthcare/Medical"
    # Financial/Business domain
    elif any(word in all_columns for word in ['price', 'cost', 'revenue', 'sales', 'profit', 'budget', 'income']):
        domain = "Business/Finance"
    # E-commerce/Retail domain
    elif any(word in all_columns for word in ['customer', 'product', 'order', 'purchase', 'item', 'cart', 'shipping']):
        domain = "Retail/E-commerce"
    # Education domain
    elif any(word in all_columns for word in ['student', 'grade', 'course', 'school', 'exam', 'score', 'teacher']):
        domain = "Education"
    # Sports domain
    elif any(word in all_columns for word in ['team', 'player', 'game', 'score', 'match', 'season', 'league', 'sport']):
        domain = "Sports"
    # Weather/Climate domain
    elif any(word in all_columns for word in ['temperature', 'weather', 'humidity', 'precipitation', 'wind', 'climate']):
        domain = "Weather/Climate"
    # Social Media domain
    elif any(word in all_columns for word in ['user', 'post', 'like', 'comment', 'share', 'follow', 'tweet', 'social']):
        domain = "Social Media"
    # Manufacturing/Industrial domain
    elif any(word in all_columns for word in ['machine', 'production', 'quality', 'defect', 'manufacturing', 'factory']):
        domain = "Manufacturing"
    # Transportation domain
    elif any(word in all_columns for word in ['vehicle', 'route', 'distance', 'speed', 'transport', 'travel', 'trip']):
        domain = "Transportation"
    
    return {
        "domain": domain,
        "analysis": f"Dataset contains {len(columns)} columns of type {data_type}. Basic analysis suggests {domain.lower()} domain.",
        "column_count": len(columns),
        "row_count": df_profile.get('n_rows', 0),
        "fallback": True
    }


# ===================== Question Generation Agent =====================
class QuestionGenerationAgent:
    def __init__(self, model: str = "gemini-1.5-flash-latest"):  # FIXED MODEL NAME
        
        self.model_name = model
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM with error handling."""
        try:
            print("🤖 Initializing Gemma LLM for question generation...")
            self.llm = GemmaLLM(
                temperature=0.5,  # Slightly higher for creative questions
                max_tokens=300
            )
            
            if self.llm.is_available():  # Check if LLM was successfully initialized
                print(f"✅ Gemma LLM initialized successfully")
            else:
                print("⚠️ Gemma LLM not available. Using fallback question generation.")
                self.llm = None
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemma LLM: {e}. Using fallback question generation.")
            self.llm = None

    def _json_serialize(self, obj):
        """Helper to serialize numpy objects."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

    def generate(self, profile: dict, understanding: dict, num_questions: int = 10):
        """
        Generate insightful analytical questions based on profile + understanding.
        For text documents, generates content-based questions.
        For structured data, generates column-based questions.
        Returns a list of questions.
        """
        print(f"🧠 Starting question generation for {num_questions} questions...")
        print(f"📊 Profile data type: {profile.get('data_type', 'unknown')}")
        print(f"🎯 Domain: {understanding.get('domain', 'Unknown')}")
        
        if self.llm is None:
            print("⚠️ No LLM available, using fallback questions")
            questions = self._fallback_questions(profile, num_questions)
            print(f"✅ Fallback generated {len(questions)} questions")
            return questions

        data_type = profile.get('data_type', 'unknown')
        
        try:
            if data_type in ['pdf', 'text']:
                print("📄 Generating text document questions...")
                questions = self._generate_text_questions(profile, understanding, num_questions)
            else:
                print("📋 Generating structured data questions...")
                questions = self._generate_structured_questions(profile, understanding, num_questions)
            
            print(f"🎯 LLM generation completed with {len(questions)} questions")
            
            # Validate questions
            if not questions or len(questions) == 0:
                print("⚠️ LLM generated 0 questions, falling back to manual generation")
                questions = self._fallback_questions(profile, num_questions)
                print(f"✅ Fallback generated {len(questions)} questions")
            
            return questions
            
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            print("🔄 Falling back to manual question generation...")
            questions = self._fallback_questions(profile, num_questions)
            print(f"✅ Fallback generated {len(questions)} questions")
            return questions

    def _generate_text_questions(self, profile: dict, understanding: dict, num_questions: int) -> list:
        """
        Generate questions for text documents based on actual content.
        """
        try:
            # Get sample content from the document
            sample_content = ""
            if 'sample' in profile and profile['sample']:
                for sample in profile['sample'][:2]:  # Use first 2 samples
                    if isinstance(sample, dict) and 'content' in sample:
                        sample_content += sample['content'][:400] + " "
            
            # Truncate for LLM
            sample_content = sample_content[:800]
            
            domain = understanding.get('domain', 'General')
            
            prompt = f"""
            Generate analytical questions for this document:
            
            Document Domain: {domain}
            Document Type: {profile.get('data_type', 'text document')}
            Chunks: {profile.get('n_rows', 'Unknown')}
            
            Sample Content:
            {sample_content}
            
            Generate {num_questions} insightful questions that would help analyze this document.
            Focus on:
            - Main topics and themes
            - Key concepts and findings
            - Insights and conclusions
            - Practical applications
            
            Questions:
            
            Questions:
            """
            
            print(f"🔄 Sending prompt to LLM (length: {len(prompt)} chars)")
            response_text = self.llm(prompt)
            
            if not response_text or len(response_text.strip()) == 0:
                print("⚠️ LLM returned empty response!")
                return self._fallback_text_questions(profile, sample_content, num_questions)
            
            print(f"✅ LLM response received (length: {len(response_text)} chars)")
            
            # Extract questions from response
            questions = self._parse_questions(response_text, num_questions)
            
            if len(questions) < num_questions:
                # Supplement with content-aware fallback questions
                fallback_questions = self._fallback_text_questions(profile, sample_content, num_questions - len(questions))
                questions.extend(fallback_questions)
            
            return questions[:num_questions]

        except Exception as e:
            print(f"⚠️ Text question generation failed: {e}. Using fallback questions.")
            return self._fallback_text_questions(profile, sample_content, num_questions)

    def _generate_structured_questions(self, profile: dict, understanding: dict, num_questions: int) -> list:
        """
        Generate questions for structured data based on columns.
        """
        try:
            # Simplified prompt for better compatibility
            column_info = []
            if 'columns' in profile and isinstance(profile['columns'], list):
                for col in profile['columns'][:5]:  # Limit to first 5 columns
                    if isinstance(col, dict):
                        column_info.append(f"- {col.get('name', 'Unknown')}: {col.get('dtype', 'Unknown type')}")
            
            # More direct prompt that works better with FLAN-T5
            prompt = f"""Analyze this dataset and generate {num_questions} questions:

Dataset: {profile.get('n_rows', 0)} rows, {profile.get('n_cols', 0)} columns
Columns: {', '.join([col.get('name', 'Unknown') for col in profile.get('columns', [])[:5] if isinstance(col, dict)])}

Generate questions about:
1. Data patterns and distributions
2. Variable relationships  
3. Outliers and anomalies
4. Predictive insights
5. Business/domain applications

List {num_questions} analytical questions (numbered 1, 2, 3...):"""
            
            print(f"🔄 Sending structured data prompt to LLM (length: {len(prompt)} chars)")
            response_text = self.llm(prompt)
            
            if not response_text or len(response_text.strip()) == 0:
                print("⚠️ LLM returned empty response for structured data!")
                return self._fallback_questions(profile, num_questions)
            
            print(f"✅ Structured data LLM response received (length: {len(response_text)} chars)")
            
            # Extract questions from response
            questions = self._parse_questions(response_text, num_questions)
            
            if len(questions) < num_questions:
                # Supplement with fallback questions
                fallback_questions = self._fallback_questions(profile, num_questions - len(questions))
                questions.extend(fallback_questions)
            
            return questions[:num_questions]

        except Exception as e:
            print(f"⚠️ Structured question generation failed: {e}. Using fallback questions.")
            return self._fallback_questions(profile, num_questions)

    def _parse_questions(self, response_text: str, num_questions: int) -> list:
        """Extract questions from LLM response with improved parsing."""
        questions = []
        
        print(f"🔍 Parsing LLM response for questions...")
        print(f"📝 Raw response length: {len(response_text)} chars")
        print(f"📝 First 200 chars: {response_text[:200]}...")
        
        # Enhanced patterns for question extraction
        patterns = [
            r'\d+\.\s*(.+?\?)',  # 1. Question?
            r'(?:^|\n)\s*([^?\n]{10,}\?)',  # Any line ending with ? (at least 10 chars)
            r'(?:Question|Q)\s*\d*:?\s*(.+?\?)',  # Question: text?
            r'(?:^|\n)\s*[-•*]\s*(.+?\?)',  # Bullet points with ?
            r'(?:^|\n)([A-Z][^?\n]{10,}\?)',  # Lines starting with capital, ending with ?
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            print(f"🔍 Pattern {i+1} found {len(matches)} matches")
            
            for match in matches:
                clean_question = match.strip()
                # Remove leading numbers, bullets, etc.
                clean_question = re.sub(r'^\d+\.\s*|^[-•*]\s*', '', clean_question)
                
                if len(clean_question) > 10 and clean_question not in questions:
                    questions.append(clean_question)
                    print(f"✅ Added question: {clean_question[:60]}...")
                    if len(questions) >= num_questions:
                        break
            if len(questions) >= num_questions:
                break
        
        # If no questions found with patterns, try simple line-by-line extraction
        if not questions:
            print("⚠️ No pattern matches. Trying line-by-line extraction...")
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10 and ('?' in line or any(word in line.lower() for word in ['what', 'how', 'why', 'which', 'where', 'when'])):
                    # Clean up the line
                    clean_line = re.sub(r'^\d+\.\s*|^[-•*]\s*', '', line)
                    if clean_line not in questions:
                        questions.append(clean_line if clean_line.endswith('?') else clean_line + '?')
                        print(f"✅ Added from line: {clean_line[:60]}...")
                        if len(questions) >= num_questions:
                            break
        
        print(f"📊 Final parsed questions count: {len(questions)}")
        return questions

    def _fallback_text_questions(self, profile: dict, content: str, num_questions: int) -> list:
        """
        Generate universal questions for ANY text document when LLM is not available.
        These questions work for any type of document regardless of domain.
        """
        # Universal question templates that work for ANY document
        universal_templates = [
            "What are the main topics discussed in this document?",
            "What are the key concepts or ideas presented?",
            "What is the primary purpose of this document?",
            "What insights can be drawn from this text?",
            "What conclusions or findings are presented?",
            "What recommendations or suggestions are provided?",
            "What problems or challenges are identified?",
            "What solutions or approaches are mentioned?",
            "What evidence or examples support the main points?",
            "What are the practical applications of the content?",
            "Who is the intended audience for this document?",
            "What background knowledge is assumed?",
            "How is the information organized or structured?",
            "What are the key takeaways from this document?",
            "What future directions or implications are discussed?",
            "What limitations or considerations are mentioned?",
            "What methodology or approach is described?",
            "What trends or patterns are identified?",
            "What terminology or concepts are defined?",
            "What references or sources are cited?"
        ]
        
        # Select questions based on document characteristics
        questions = []
        
        # Always include the most universal questions first
        priority_questions = [
            "What are the main topics discussed in this document?",
            "What are the key concepts or ideas presented?",
            "What is the primary purpose of this document?",
            "What insights can be drawn from this text?",
            "What conclusions or findings are presented?"
        ]
        
        # Add priority questions first
        for question in priority_questions:
            if len(questions) < num_questions:
                questions.append(question)
        
        # Add remaining universal questions
        for template in universal_templates:
            if len(questions) >= num_questions:
                break
            if template not in questions:  # Avoid duplicates
                questions.append(template)
        
        return questions[:num_questions]

    def _fallback_questions(self, profile: dict, num_questions: int) -> list:
        """Generate domain-aware questions when LLM is not available."""
        questions = []
        columns = profile.get('columns', [])
        
        # Get domain from profile or detect it
        domain = "General"
        if 'domain' in profile:
            domain = profile['domain']
        else:
            # Quick domain detection
            column_names = [col.get('name', str(col)).lower() if isinstance(col, dict) else str(col).lower() for col in columns]
            all_columns_text = ' '.join(column_names)
            
            if any(word in all_columns_text for word in ['sleep', 'duration', 'quality']):
                domain = "Sleep/Health"
            elif any(word in all_columns_text for word in ['stress', 'bmi', 'blood pressure', 'heart rate']):
                domain = "Healthcare/Medical"
        
        # Domain-specific question templates
        if domain in ["Sleep/Health", "Healthcare/Medical"]:
            domain_questions = [
                "What factors most strongly correlate with sleep quality?",
                "How does stress level impact sleep duration and quality?",
                "What are the optimal ranges for health metrics in this dataset?", 
                "Which demographic groups show the best sleep patterns?",
                "Are there any sleep disorders that correlate with specific health indicators?",
                "How do lifestyle factors (activity, BMI) affect sleep outcomes?",
                "What patterns exist between age, occupation and sleep quality?",
                "Can we identify risk factors for poor sleep health?",
                "How do physical health metrics relate to sleep disorders?",
                "What insights can help improve overall sleep and health outcomes?"
            ]
        else:
            # Generic analytical questions for any dataset
            domain_questions = [
                "What are the key patterns and trends in this dataset?",
                "Which variables show the strongest correlations?",
                "Are there any significant outliers or anomalies?",
                "How do different segments compare across key metrics?",
                "What predictive insights can be derived from this data?",
                "Which factors are most important for decision making?",
                "Are there any data quality issues that need attention?",
                "What time-based trends exist in the data?",
                "How can this data be used to optimize outcomes?",
                "What actionable insights emerge from the analysis?"
            ]
        
        # Use domain-specific questions first
        questions.extend(domain_questions[:num_questions])
        
        # If we need more questions, add column-specific ones
        if len(questions) < num_questions:
            column_templates = [
                "What is the distribution of values in {}?",
                "How does {} correlate with other key variables?",
                "Are there any outliers or unusual patterns in {}?",
                "What insights can be derived from {} analysis?"
            ]
        
            # Generate column-specific questions if needed
            for col in columns:
                if len(questions) >= num_questions:
                    break
                    
                if isinstance(col, dict):
                    col_name = col.get('name', 'unknown_column')
                else:
                    col_name = str(col)
                
                if len(column_templates) > 0:
                    template_idx = (len(questions) - len(domain_questions)) % len(column_templates)
                    template = column_templates[template_idx]
                    questions.append(template.format(col_name))
            
        return questions[:num_questions]


# ===================== Stateful Wrapper =====================
def question_generation_node(dataset_name: str, df_profile: dict, num_questions: int = 10):
    """Generate understanding + questions for dataset and update WorkflowState."""
    print(f"🔍 QuestionGen - Starting for dataset: {dataset_name}")
    
    # Get data understanding
    understanding = data_understanding(df_profile)
    print(f"🔍 QuestionGen - Understanding: {understanding.get('domain', 'Unknown')}")

    # Generate questions
    agent = QuestionGenerationAgent()
    questions = agent.generate(df_profile, understanding, num_questions)
    
    print(f"🔍 QuestionGen - Agent returned {len(questions)} questions")
    print(f"🔍 QuestionGen - Questions type: {type(questions)}")
    print(f"🔍 QuestionGen - First question: {questions[0] if questions else 'NONE'}")

    # Update global state
    STATE.understanding[dataset_name] = understanding
    STATE.questions[dataset_name] = questions
    
    # Verify the update
    stored_questions = STATE.questions.get(dataset_name, [])
    print(f"🔍 QuestionGen - Stored in STATE: {len(stored_questions)} questions")

    print(f"\n✅ Generated Questions for '{dataset_name}':")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    return STATE
