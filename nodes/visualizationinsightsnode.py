import os
import re
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from dotenv import load_dotenv
from gemma_llm import GemmaLLM

# Structured + Text plotting
from utils.plotting_utils import (
    # Structured
    plot_histogram, plot_scatter, plot_bar, plot_line,
    plot_box, plot_violin, plot_heatmap, plot_correlation_matrix, plot_timeseries,
    # Text
    plot_word_frequencies, plot_wordcloud, sentiment_distribution
)

# Import WorkflowState if you're using a global state
from utils.state import STATE

load_dotenv()


class InsightAgent:
    def __init__(self, model: str = "gemini-1.5-flash-latest"):
        self.model_name = model
        self.llm = None
        self.dataset_cache = {}       # ✅ cache per dataset
        self.analysis_history = {}    # ✅ history per dataset
        self.text_cache = {}          # ✅ text datasets (e.g., pdf, txt)
        self._init_llm()

    # ================= INIT LLM =================
    def _init_llm(self):
        try:
            # Initialize Gemma LLM for superior analytical capabilities
            print("🤖 Initializing Gemma LLM...")
            self.llm = GemmaLLM(
                temperature=0.3,
                max_tokens=500
            )
            
            if self.llm.is_available():  # Check if LLM was successfully initialized
                print("✅ Gemma LLM initialized successfully")
            else:
                print("⚠️ Gemma LLM not available. Using statistical analysis only.")
                self.llm = None
                
        except Exception as e:
            print(f"⚠️ Failed to init Gemma LLM ({e}). Using stats only.")
            self.llm = None

    # ================= CACHE =================
    def set_dataset(self, dataset_name: str, df: pd.DataFrame):
        """Cache DataFrame-level metrics for reuse."""
        if dataset_name not in self.dataset_cache:
            print(f"🧠 Precomputing metrics for dataset: {dataset_name}")
            # Get properly typed columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            
            # Safe correlation matrix calculation
            correlation_matrix = None
            if len(numeric_cols) > 1:
                try:
                    # Only use truly numeric columns for correlation
                    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        correlation_matrix = numeric_df.corr()
                except Exception as e:
                    print(f"⚠️ Correlation calculation failed: {e}")
                    correlation_matrix = None
            
            self.dataset_cache[dataset_name] = {
                "df": df,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "basic_stats": self._compute_basic_stats(df),
                "correlation_matrix": correlation_matrix,
            }
            self.analysis_history[dataset_name] = []

    def set_text_dataset(self, dataset_name: str, docs: list[str]):
        """Cache text data (e.g., from PDF or txt)."""
        if dataset_name not in self.text_cache:
            print(f"🧠 Caching text dataset: {dataset_name} ({len(docs)} docs)")
            self.text_cache[dataset_name] = {
                "docs": docs,
                "stats": {
                    "num_docs": len(docs),
                    "avg_len": np.mean([len(d.split()) for d in docs]) if docs else 0
                }
            }
            self.analysis_history[dataset_name] = []

    def _compute_basic_stats(self, df: pd.DataFrame) -> dict:
        stats_dict = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "q1": df[col].quantile(0.25),
                "q3": df[col].quantile(0.75),
                "count": df[col].count(),
            }
        return stats_dict

    # ================= LLM-DRIVEN INTELLIGENT ANALYSIS =================
    def _intelligent_analysis(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """Simplified LLM-driven analysis that generates proper insights and visualizations"""
        if not self.llm or not self.llm.is_available():
            return self._fallback_analysis(df, question, numeric_cols, categorical_cols)
        
        print(f"🤖 Analyzing: {question}")
        
        try:
            # Step 1: Use LLM to determine analysis type
            analysis_type = self._determine_analysis_type_llm(question, numeric_cols, categorical_cols)
            print(f"📊 Analysis type: {analysis_type}")
            
            # Step 2: Use LLM to find relevant columns for the question
            relevant_cols = self._find_relevant_columns_llm(question, numeric_cols, categorical_cols, df)
            print(f"🎯 Using columns: {relevant_cols}")
            print(f"🗓️ Available - Numeric: {numeric_cols[:3]}... Categorical: {categorical_cols[:3]}...")
            
            # Step 3: Try comprehensive LLM analysis first for custom questions
            if self.llm and self.llm.is_available():
                llm_result = self._comprehensive_llm_analysis(df, question, relevant_cols, numeric_cols, categorical_cols)
                if llm_result and llm_result.get("answer") and "error" not in llm_result["answer"].lower():
                    print(f"✅ Using comprehensive LLM analysis")
                    return llm_result
            
            # Step 4: Fallback to structured analysis
            print(f"🔄 Fallback to structured analysis")
            result = self._perform_analysis(df, question, analysis_type, relevant_cols, numeric_cols, categorical_cols)
            
            # Step 5: Generate LLM insights
            if result["answer"]:
                enhanced_insights = self._generate_llm_insights(df, question, result, relevant_cols)
                if enhanced_insights:
                    result["answer"] = enhanced_insights
            
            return result
            
        except Exception as e:
            print(f"⚠️ Intelligent analysis failed: {e}")
            return self._fallback_analysis(df, question, numeric_cols, categorical_cols)
    
    def _create_data_context(self, df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> str:
        """Create comprehensive data context for LLM"""
        context = f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        
        # Numeric columns summary
        if numeric_cols:
            context += "NUMERIC COLUMNS:\n"
            for col in numeric_cols[:5]:  # Limit to first 5
                stats = df[col].describe()
                context += f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]\n"
        
        # Categorical columns summary
        if categorical_cols:
            context += "\nCATEGORICAL COLUMNS:\n"
            for col in categorical_cols[:5]:  # Limit to first 5
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3)
                context += f"- {col}: {unique_count} unique values, top: {dict(top_values)}\n"
        
        # Sample data
        context += f"\nSAMPLE DATA (first 3 rows):\n{df.head(3).to_string()}\n"
        
        return context
    
    def _determine_analysis_type_llm(self, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Use LLM to determine what type of analysis to perform"""
        if not self.llm or not self.llm.is_available():
            # Simple fallback
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                return "comparison"
            elif len(numeric_cols) >= 2:
                return "correlation"
            elif len(numeric_cols) >= 1:
                return "distribution"
            else:
                return "aggregation"
        
        try:
            analysis_types = [
                "correlation - relationships between numeric variables",
                "distribution - patterns and spread of single variable", 
                "comparison - comparing groups or categories",
                "aggregation - summary statistics and totals",
                "trend - changes over time"
            ]
            
            prompt = f"""Question: "{question}"

Available data:
- Numeric columns: {len(numeric_cols)} ({numeric_cols[:3]}...)
- Categorical columns: {len(categorical_cols)} ({categorical_cols[:3]}...)

What type of analysis is most appropriate?

Options:
{chr(10).join(analysis_types)}

Respond with only one word: correlation, distribution, comparison, aggregation, or trend"""
            
            response = self.llm(prompt).strip().lower()
            
            # Validate response
            valid_types = ["correlation", "distribution", "comparison", "aggregation", "trend"]
            if response in valid_types:
                return response
            
            # Try to extract valid type from response
            for valid_type in valid_types:
                if valid_type in response:
                    return valid_type
            
            # Fallback
            return "comparison" if categorical_cols and numeric_cols else "distribution"
            
        except Exception as e:
            print(f"⚠️ LLM analysis type detection failed: {e}")
            return "comparison" if categorical_cols and numeric_cols else "distribution"
    
    def _comprehensive_llm_analysis(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> dict:
        """Comprehensive LLM-driven analysis for any question"""
        try:
            # Create focused data context
            context_parts = []
            context_parts.append(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Add relevant column information
            for col in relevant_cols[:3]:  # Limit to top 3 relevant columns
                if col in numeric_cols:
                    stats = df[col].describe()
                    context_parts.append(f"- {col} (numeric): mean={stats['mean']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
                elif col in categorical_cols:
                    value_counts = df[col].value_counts().head(3)
                    context_parts.append(f"- {col} (categorical): top values {dict(value_counts)}")
            
            # Add sample data
            if relevant_cols:
                sample_data = df[relevant_cols].head(3)
                context_parts.append(f"\nSample data:\n{sample_data.to_string()}")
            
            # Create focused prompt for analysis
            prompt = f"""Analyze this data to answer the question.

QUESTION: {question}

DATA CONTEXT:
{chr(10).join(context_parts)}

Provide:
1. Direct numerical answer or finding
2. Brief explanation in 1-2 sentences
3. One key insight

Be specific and quantitative. Focus on what the data actually shows."""
            
            response = self.llm(prompt)
            
            # Validate response quality
            if response and len(response.strip()) > 20 and len(response.strip()) < 800:
                # Generate appropriate visualization
                viz_html = self._suggest_visualization(df, question, numeric_cols, categorical_cols)
                
                return {
                    "question": question,
                    "answer": f"🤖 AI Analysis:\n{response.strip()}",
                    "visualization_html": viz_html,
                    "method": "comprehensive_llm_analysis"
                }
            
            return None  # Signal to use fallback
            
        except Exception as e:
            print(f"⚠️ Comprehensive LLM analysis failed: {e}")
            return None
    
    def _find_relevant_columns_llm(self, question: str, numeric_cols: list, categorical_cols: list, df: pd.DataFrame) -> list:
        """Use LLM to intelligently find relevant columns for the question"""
        if not self.llm or not self.llm.is_available():
            # Simple fallback without complex logic
            all_cols = numeric_cols + categorical_cols
            return all_cols[:3] if all_cols else []
        
        try:
            columns_info = []
            # Provide column info to LLM
            for col in numeric_cols:
                sample_vals = df[col].dropna().head(3).tolist()
                columns_info.append(f"- {col} (numeric): sample values {sample_vals}")
            
            for col in categorical_cols:
                unique_vals = df[col].value_counts().head(3).index.tolist()
                columns_info.append(f"- {col} (categorical): top values {unique_vals}")
            
            prompt = f"""Given this question: "{question}"

Which columns from this dataset are most relevant? Choose 2-3 columns:

{chr(10).join(columns_info[:10])}

Respond with only the column names, separated by commas (e.g: column1, column2, column3)"""
            
            response = self.llm(prompt).strip()
            
            # Parse LLM response
            suggested_cols = [col.strip() for col in response.split(',')]
            all_cols = numeric_cols + categorical_cols
            
            # Validate and return existing columns
            relevant = [col for col in suggested_cols if col in all_cols]
            
            # Fallback if LLM response is poor
            if not relevant:
                return all_cols[:3] if all_cols else []
            
            return relevant[:3]  # Limit to 3 columns
            
        except Exception as e:
            print(f"⚠️ LLM column selection failed: {e}")
            all_cols = numeric_cols + categorical_cols
            return all_cols[:3] if all_cols else []
    
    def _perform_analysis(self, df: pd.DataFrame, question: str, analysis_type: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> dict:
        """Perform the actual analysis based on type and columns"""
        result = {
            "question": question,
            "answer": "",
            "visualization_html": None,
            "method": "intelligent_analysis"
        }
        
        try:
            if analysis_type == "correlation" and len([col for col in relevant_cols if col in numeric_cols]) >= 2:
                numeric_relevant = [col for col in relevant_cols if col in numeric_cols]
                corr_matrix = df[numeric_relevant].corr()
                result["answer"] = self._generate_correlation_insights(corr_matrix, question)
                # Use intelligent visualization decision
                if self._should_create_visualization(question):
                    result["visualization_html"] = plot_correlation_matrix(df[numeric_relevant])
                
            elif analysis_type == "distribution":
                target_col = next((col for col in relevant_cols if col in numeric_cols), numeric_cols[0] if numeric_cols else None)
                if target_col:
                    result["answer"] = self._generate_distribution_insights(df[target_col], target_col, question)
                    # Use intelligent visualization decision
                    if self._should_create_visualization(question):
                        result["visualization_html"] = plot_histogram(df, target_col)
                    
            elif analysis_type == "comparison":
                cat_col = next((col for col in relevant_cols if col in categorical_cols), categorical_cols[0] if categorical_cols else None)
                num_col = next((col for col in relevant_cols if col in numeric_cols), numeric_cols[0] if numeric_cols else None)
                if cat_col and num_col:
                    grouped = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).reset_index()
                    result["answer"] = self._generate_comparison_insights(grouped, cat_col, num_col, question)
                    # Use intelligent visualization decision
                    if self._should_create_visualization(question):
                        result["visualization_html"] = plot_bar(df.groupby(cat_col)[num_col].mean().reset_index(), cat_col, num_col)
                    
            elif analysis_type == "aggregation":
                target_col = next((col for col in relevant_cols if col in numeric_cols), numeric_cols[0] if numeric_cols else None)
                if target_col:
                    stats = df[target_col].describe()
                    result["answer"] = f"📊 Summary statistics for {target_col}:\n" + \
                                     f"• Mean: {stats['mean']:.2f}\n" + \
                                     f"• Median: {stats['50%']:.2f}\n" + \
                                     f"• Std Dev: {stats['std']:.2f}\n" + \
                                     f"• Range: {stats['min']:.2f} to {stats['max']:.2f}"
                    # Use intelligent visualization decision - aggregation usually doesn't need viz
                    if self._should_create_visualization(question):
                        result["visualization_html"] = plot_histogram(df, target_col)
                    
            else:
                # Fallback to basic analysis
                if relevant_cols and relevant_cols[0] in numeric_cols:
                    target_col = relevant_cols[0]
                    result["answer"] = f"📊 Basic analysis of {target_col}:\n" + \
                                     f"• Average: {df[target_col].mean():.2f}\n" + \
                                     f"• Total records: {len(df)}\n" + \
                                     f"• Missing values: {df[target_col].isnull().sum()}"
                    # Use intelligent visualization decision for fallback too
                    if self._should_create_visualization(question):
                        result["visualization_html"] = plot_histogram(df, target_col)
                    
        except Exception as e:
            print(f"⚠️ Analysis execution failed: {e}")
            result["answer"] = f"Could not perform {analysis_type} analysis: {str(e)}"
            
        return result
    
    def _generate_llm_insights(self, df: pd.DataFrame, question: str, result: dict, relevant_cols: list) -> str:
        """Use LLM to enhance insights with natural language understanding"""
        try:
            # Simplified approach to avoid garbled output
            # Just use the statistical results we already generated
            base_answer = result.get("answer", "")
            
            # Only enhance if we have a good base answer and LLM is working
            if not base_answer or "could not perform" in base_answer.lower():
                return base_answer
            
            # Create a very simple, focused prompt
            simple_prompt = f"""Explain this data analysis result in simple terms:

Question: {question}
Result: {base_answer[:200]}

Write a brief explanation in 2-3 sentences about what this means."""
            
            llm_response = self.llm(simple_prompt)
            
            # Strict validation of LLM response
            if (llm_response and 
                len(llm_response.strip()) > 10 and 
                len(llm_response.strip()) < 500 and
                not any(char.isdigit() and llm_response.count(char) > 10 for char in "0123456789")):
                
                # Clean response
                clean_response = llm_response.strip()
                # Remove any data-like patterns (sequences of numbers)
                import re
                if not re.search(r'\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+', clean_response):
                    return f"{base_answer}\n\n💡 Insight: {clean_response}"
            
            # Fallback to base answer if LLM response looks garbled
            print(f"🔄 Using statistical analysis (LLM response filtered out)")
            return base_answer
                
        except Exception as e:
            print(f"⚠️ LLM insight generation failed: {e}")
            return result.get("answer", "Analysis could not be completed")
    
    
    def _generate_correlation_insights(self, corr_matrix, question: str) -> str:
        """Generate intelligent correlation insights"""
        strong_corr = corr_matrix.abs() > 0.5
        insights = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j and strong_corr.iloc[i, j]:
                    corr_val = corr_matrix.iloc[i, j]
                    strength = "strong" if abs(corr_val) > 0.7 else "moderate"
                    direction = "positive" if corr_val > 0 else "negative"
                    insights.append(f"{col1} has a {strength} {direction} correlation ({corr_val:.3f}) with {col2}")
        
        if insights:
            return "📊 Correlation Analysis:\n" + "\n".join(insights[:3])
        else:
            return "📊 No strong correlations found between variables."
    
    def _generate_distribution_insights(self, series: pd.Series, col_name: str, question: str) -> str:
        """Generate intelligent distribution insights"""
        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        skewness = series.skew()
        
        insights = [f"📈 Distribution Analysis for {col_name}:"]
        insights.append(f"• Mean: {mean_val:.2f}, Median: {median_val:.2f}")
        
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"• Highly skewed to the {direction} (skewness: {skewness:.2f})")
        elif abs(skewness) > 0.5:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"• Moderately skewed to the {direction}")
        else:
            insights.append("• Approximately normal distribution")
        
        # Outliers
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            insights.append(f"• Contains {len(outliers)} outliers ({len(outliers)/len(series)*100:.1f}% of data)")
        
        return "\n".join(insights)
    
    def _generate_comparison_insights(self, grouped_data, cat_col: str, num_col: str, question: str) -> str:
        """Generate intelligent comparison insights"""
        insights = [f"📊 Comparison Analysis: {num_col} by {cat_col}"]
        
        # Sort by mean value
        sorted_data = grouped_data.sort_values('mean', ascending=False)
        
        highest = sorted_data.iloc[0]
        lowest = sorted_data.iloc[-1]
        
        insights.append(f"• Highest average: {highest[cat_col]} ({highest['mean']:.2f})")
        insights.append(f"• Lowest average: {lowest[cat_col]} ({lowest['mean']:.2f})")
        
        # Variation analysis
        overall_std = grouped_data['mean'].std()
        if overall_std > grouped_data['mean'].mean() * 0.1:  # 10% threshold
            insights.append(f"• Significant variation across {cat_col} groups")
        else:
            insights.append(f"• Relatively consistent across {cat_col} groups")
        
        return "\n".join(insights)
    
    def _custom_llm_analysis(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """Let LLM perform custom analysis when standard patterns don't apply"""
        if not self.llm:
            return self._fallback_analysis(df, question, numeric_cols, categorical_cols)
        
        # Create rich context for LLM
        context = f"""
        QUESTION: {question}
        
        DATA SUMMARY:
        - Shape: {df.shape}
        - Numeric columns: {numeric_cols[:5]}
        - Categorical columns: {categorical_cols[:5]}
        
        KEY STATISTICS:
        {df.describe().to_string() if numeric_cols else 'No numeric columns'}
        
        SAMPLE DATA:
        {df.head(3).to_string()}
        """
        
        analysis_prompt = f"""
        You are an expert data analyst. Analyze this dataset to answer the user's question.
        
        {context}
        
        Provide:
        1. Direct answer to the question with specific numbers/findings
        2. Key insights and patterns you discovered
        3. Any notable observations or recommendations
        
        Be specific, quantitative, and actionable. Focus on what the data actually shows.
        """
        
        try:
            llm_response = self.llm(analysis_prompt)
            return {
                "question": question,
                "answer": f"🤖 AI Analysis:\n{llm_response}",
                "visualization_html": self._suggest_visualization(df, question, numeric_cols, categorical_cols),
                "method": "custom_llm_analysis"
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Custom analysis failed: {e}",
                "visualization_html": None,
                "method": "error"
            }
    
    def _suggest_visualization(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Use LLM to intelligently decide if visualization is needed and what type"""
        try:
            # First, ask LLM if visualization is even needed
            if not self._should_create_visualization(question):
                print(f"🧠 LLM decided: No visualization needed for this question")
                return None
            
            # If visualization is needed, determine the best type
            viz_type = self._determine_visualization_type(question, numeric_cols, categorical_cols)
            print(f"📊 LLM suggested visualization: {viz_type}")
            
            # Create the appropriate visualization
            return self._create_visualization(df, viz_type, question, numeric_cols, categorical_cols)
            
        except Exception as e:
            print(f"⚠️ Intelligent visualization creation failed: {e}")
            return None
    
    def _should_create_visualization(self, question: str) -> bool:
        """Use LLM to determine if this question needs a visualization"""
        if not self.llm or not self.llm.is_available():
            # Simple fallback - only create viz for obvious visual questions
            q_lower = question.lower()
            visual_keywords = ['show', 'plot', 'chart', 'graph', 'visualize', 'distribution', 
                             'correlation', 'trend', 'pattern', 'compare', 'comparison']
            return any(keyword in q_lower for keyword in visual_keywords)
        
        try:
            prompt = f"""Does this question require a chart, graph, or visualization to answer properly?

Question: "{question}"

Some questions need visual answers (like "show me the distribution", "what's the trend?", "compare sales by region"), while others just need text answers (like "how many rows?", "what's the average?", "list the columns").

Answer only: YES or NO"""
            
            response = self.llm(prompt).strip().upper()
            
            # Parse LLM response
            if 'YES' in response:
                return True
            elif 'NO' in response:
                return False
            else:
                # Fallback to keyword detection if LLM response is unclear
                q_lower = question.lower()
                visual_keywords = ['show', 'plot', 'chart', 'graph', 'visualize', 'distribution', 
                                 'correlation', 'trend', 'pattern', 'compare', 'comparison']
                return any(keyword in q_lower for keyword in visual_keywords)
                
        except Exception as e:
            print(f"⚠️ LLM visualization decision failed: {e}")
            # Safe fallback
            return False
    
    def _determine_visualization_type(self, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Use LLM to determine the best visualization type"""
        if not self.llm or not self.llm.is_available():
            # Simple fallback logic
            q_lower = question.lower()
            if "correlation" in q_lower and len(numeric_cols) >= 2:
                return "correlation"
            elif "distribution" in q_lower and numeric_cols:
                return "distribution"
            elif ("compare" in q_lower or "difference" in q_lower) and categorical_cols and numeric_cols:
                return "comparison"
            elif numeric_cols:
                return "distribution"
            else:
                return "summary"
        
        try:
            viz_options = [
                "histogram - for showing distribution of a single numeric variable",
                "bar_chart - for comparing values across categories", 
                "correlation_matrix - for showing relationships between multiple numeric variables",
                "scatter_plot - for showing relationship between two numeric variables",
                "line_chart - for showing trends over time",
                "summary - for basic statistics that don't need a chart"
            ]
            
            prompt = f"""What type of visualization best answers this question?

Question: "{question}"

Available data:
- Numeric columns: {numeric_cols[:5] if numeric_cols else 'None'}
- Categorical columns: {categorical_cols[:5] if categorical_cols else 'None'}

Visualization options:
{chr(10).join(viz_options)}

Choose the SINGLE best option. Respond with just the option name (e.g., "histogram", "bar_chart", etc.)"""
            
            response = self.llm(prompt).strip().lower()
            
            # Map LLM response to our visualization types
            viz_mapping = {
                'histogram': 'distribution',
                'bar_chart': 'comparison', 
                'bar': 'comparison',
                'correlation_matrix': 'correlation',
                'correlation': 'correlation',
                'scatter_plot': 'scatter',
                'scatter': 'scatter',
                'line_chart': 'trend',
                'line': 'trend',
                'summary': 'summary'
            }
            
            for key, value in viz_mapping.items():
                if key in response:
                    return value
            
            # Fallback
            return 'distribution' if numeric_cols else 'summary'
            
        except Exception as e:
            print(f"⚠️ LLM visualization type detection failed: {e}")
            return 'distribution' if numeric_cols else 'summary'
    
    def _create_visualization(self, df: pd.DataFrame, viz_type: str, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Create the specific visualization based on the determined type"""
        try:
            if viz_type == 'correlation' and len(numeric_cols) >= 2:
                return plot_correlation_matrix(df[numeric_cols].select_dtypes(include=[np.number]))
                
            elif viz_type == 'distribution' and numeric_cols:
                # Find most relevant column using LLM or simple matching
                target_col = self._find_target_column(question, numeric_cols)
                return plot_histogram(df, target_col)
                
            elif viz_type == 'comparison' and categorical_cols and numeric_cols:
                cat_col = self._find_target_column(question, categorical_cols)
                num_col = self._find_target_column(question, numeric_cols)
                return plot_bar(df.groupby(cat_col)[num_col].mean().reset_index(), cat_col, num_col)
                
            elif viz_type == 'scatter' and len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                return plot_scatter(df, col1, col2)
                
            elif viz_type == 'trend' and numeric_cols:
                # Assume first numeric column is the value, look for time-like columns
                target_col = numeric_cols[0]
                time_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'year', 'month'])]
                if time_cols:
                    return plot_line(df, time_cols[0], target_col)
                else:
                    return plot_line(df, df.index.name or 'index', target_col)
                    
            else:
                # viz_type == 'summary' or fallback - no visualization
                return None
                
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
            return None
    
    def _find_target_column(self, question: str, columns: list) -> str:
        """Find the most relevant column for the question"""
        if not columns:
            return None
        
        q_lower = question.lower()
        
        # Look for column names mentioned in the question
        for col in columns:
            if col.lower() in q_lower:
                return col
        
        # Return first column as fallback
        return columns[0]
    
    def _fallback_analysis(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """Fallback analysis when LLM is not available"""
        # Even in fallback, use simple keyword-based visualization decision
        viz_html = None
        if numeric_cols and self._should_create_visualization(question):
            viz_html = plot_histogram(df, numeric_cols[0])
        
        return {
            "question": question,
            "answer": f"Basic analysis: Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Available for analysis: {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.",
            "visualization_html": viz_html,
            "method": "fallback_analysis"
        }

    # Old methods removed - now using intelligent LLM-driven analysis

    # ================= MAIN ENTRY =================
    def answer(self, data, question: str, dataset_name: str = None) -> dict:
        """
        Unified entrypoint:
        - Structured (DataFrame)
        - Text (list[str])
        """
        if isinstance(data, pd.DataFrame):
            return self._answer_structured(data, question, dataset_name)
        elif isinstance(data, list) and all(isinstance(d, str) for d in data):
            return self._answer_text(data, question, dataset_name)
        else:
            return {
                "answer": "⚠️ Unsupported data type. Provide DataFrame or list[str].",
                "visualization_html": None,
            }

    # ================= STRUCTURED =================
    def _answer_structured(self, df: pd.DataFrame, question: str, dataset_name: str = None) -> dict:
        # Ensure dataset is cached for performance
        if dataset_name and dataset_name not in self.dataset_cache:
            self.set_dataset(dataset_name, df)
        
        # Get column information
        if dataset_name and dataset_name in self.dataset_cache:
            cache = self.dataset_cache[dataset_name]
            numeric_cols, categorical_cols = cache["numeric_cols"], cache["categorical_cols"]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 🚀 Use intelligent LLM-driven analysis instead of primitive keyword matching
        analysis_result = self._intelligent_analysis(df, question, numeric_cols, categorical_cols)

        # Store in history
        if dataset_name and dataset_name in self.analysis_history:
            self.analysis_history[dataset_name].append({
                "question": question,
                "result": analysis_result,
                "timestamp": pd.Timestamp.now(),
            })

        return analysis_result

    # ================= TEXT =================
    def _answer_text(self, docs: list[str], question: str, dataset_name: str = None) -> dict:
        q_lower = question.lower()
        answer, viz_html = f"📝 Text Analysis for: '{question}'\n", None

        try:
            if any(word in q_lower for word in ["word", "keyword", "frequency"]):
                viz_html = plot_word_frequencies(docs)
                answer += "📊 Word frequency plot generated."
            elif any(word in q_lower for word in ["cloud", "overview", "summary"]):
                viz_html = plot_wordcloud(docs)
                answer += "☁️ Word cloud generated."
            elif any(word in q_lower for word in ["sentiment", "tone", "positive", "negative"]):
                viz_html = sentiment_distribution(docs)
                answer += "😊 Sentiment distribution generated."
            else:
                stats = {
                    "num_docs": len(docs),
                    "avg_len": np.mean([len(d.split()) for d in docs]) if docs else 0
                }
                answer += f"Corpus has {stats['num_docs']} docs, avg length {stats['avg_len']:.2f} words."
                viz_html = plot_word_frequencies(docs)
        except Exception as e:
            answer += f"\n⚠️ Error during text analysis: {e}"

        result = {
            "question": question,
            "answer": answer,
            "visualization_html": viz_html,
            "method": "text_analysis",
        }

        if dataset_name and dataset_name in self.analysis_history:
            self.analysis_history[dataset_name].append({
                "question": question,
                "result": result,
                "timestamp": pd.Timestamp.now(),
            })

        return result

    # ================= MULTI-Q =================
    def answer_multiple(self, state: STATE, data, questions: list, dataset_name: str) -> STATE:
        results = []
        print(f"\n🔄 Processing {len(questions)} questions...")

        # precompute structured or text cache
        if isinstance(data, pd.DataFrame) and dataset_name and dataset_name not in self.dataset_cache:
            self.set_dataset(dataset_name, data)
        elif isinstance(data, list) and dataset_name and dataset_name not in self.text_cache:
            self.set_text_dataset(dataset_name, data)

        for i, question in enumerate(questions, 1):
            print(f"   Q{i}/{len(questions)}: {question[:50]}...")
            try:
                result = self.answer(data, question, dataset_name)
                results.append(result)
            except Exception as e:
                print(f"   ⚠️ Failed Q{i}: {e}")
                results.append({
                    "question": question,
                    "answer": f"⚠️ Error processing question: {e}",
                    "visualization_html": None,
                    "method": "error"
                })

        print(f"✅ Completed {len(results)} questions")

        # update state
        state.insights[dataset_name] = results
        return state