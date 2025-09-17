from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import os
import glob
import re
from datetime import datetime
import tempfile
from utils.state import STATE


def generate_enhanced_report(state: STATE, dataset_name: str, output_path="enhanced_report.pdf", analysis_history=None) -> str:
    """
    Generate a comprehensive PDF report with dataset profile, insights, and embedded visualizations.
    """
    
    profile = state.profiles.get(dataset_name, {})
    insights = state.insights.get(dataset_name, [])
    questions = state.questions.get(dataset_name, [])
    understanding = state.understanding.get(dataset_name, {})
    
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        textColor=colors.darkgreen
    )
    
    story = []
    
    # Title Page
    story.append(Paragraph(f"📊 Enhanced Data Analysis Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Dataset: {dataset_name.replace('_', ' ').title()}", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Domain: {understanding.get('domain', 'General')}", styles['Normal']))
    story.append(Spacer(1, 40))
    
    # Executive Summary
    story.append(Paragraph("📋 Executive Summary", subtitle_style))
    if dataset_name in state.datasets:
        df = state.datasets[dataset_name]
        numeric_cols = df.select_dtypes(include='number').columns
        categorical_cols = df.select_dtypes(include='object').columns
        
        summary_text = f"""
        This report presents a comprehensive analysis of the {dataset_name.replace('_', ' ')} dataset. 
        The dataset contains {df.shape[0]:,} records with {df.shape[1]} features, including 
        {len(numeric_cols)} numeric and {len(categorical_cols)} categorical variables. 
        The analysis identified this as a {understanding.get('domain', 'general')} domain dataset.
        """
        story.append(Paragraph(summary_text.strip(), styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Dataset Overview
    story.append(Paragraph("📊 Dataset Overview", subtitle_style))
    if dataset_name in state.datasets:
        df = state.datasets[dataset_name]
        story.append(Paragraph(f"• Total Records: {df.shape[0]:,}", styles['Normal']))
        story.append(Paragraph(f"• Total Features: {df.shape[1]}", styles['Normal']))
        story.append(Paragraph(f"• Numeric Features: {len(df.select_dtypes(include='number').columns)}", styles['Normal']))
        story.append(Paragraph(f"• Categorical Features: {len(df.select_dtypes(include='object').columns)}", styles['Normal']))
        story.append(Paragraph(f"• Missing Values: {df.isnull().sum().sum():,}", styles['Normal']))
        story.append(Paragraph(f"• Duplicate Records: {df.duplicated().sum():,}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Column Details Table
    if "columns" in profile and isinstance(profile["columns"], list) and all(isinstance(c, dict) for c in profile["columns"]):
        story.append(Paragraph("🔍 Feature Analysis", subtitle_style))
        df_profile = pd.DataFrame(profile["columns"])
        
        # Select key columns for the table
        key_columns = ['name', 'dtype', 'num_missing', 'num_unique']
        if 'mean' in df_profile.columns:
            key_columns.append('mean')
        if 'std' in df_profile.columns:
            key_columns.append('std')
            
        display_df = df_profile[key_columns].head(10)  # Show first 10 columns
        table_data = [display_df.columns.tolist()] + display_df.values.tolist()
        
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
    
    # Generated Questions
    if questions:
        story.append(Paragraph("❓ Generated Analysis Questions", subtitle_style))
        story.append(Paragraph("The following questions were automatically generated to guide the analysis:", styles['Normal']))
        story.append(Spacer(1, 10))
        
        for i, question in enumerate(questions[:8], 1):  # Show first 8 questions
            clean_question = question.replace('*', '').replace('**', '').strip()
            story.append(Paragraph(f"{i}. {clean_question}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # User Analysis Sessions - THE MISSING PIECE!
    if analysis_history and dataset_name in analysis_history:
        user_sessions = analysis_history[dataset_name]
        if user_sessions:
            story.append(Paragraph("🔍 User Analysis Sessions", subtitle_style))
            story.append(Paragraph(f"The following {len(user_sessions)} questions were asked by the user with AI-generated insights:", styles['Normal']))
            story.append(Spacer(1, 15))
            
            for i, session in enumerate(user_sessions, 1):
                # Session header with timestamp
                timestamp = datetime.strptime(session['timestamp'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d %H:%M')
                
                # Question
                session_question = session.get('original_input', session.get('question', 'Unknown question'))
                story.append(Paragraph(f"Q{i}. [{timestamp}] {session_question}", styles['Heading3']))
                
                # Answer with context info
                context_info = ""
                if session.get('context_used', 0) > 0:
                    context_info = f" (Used {session['context_used']} previous insights)"
                
                answer_text = session.get('answer', 'No answer available')
                # Clean up the answer text for PDF
                clean_answer = re.sub(r'[📊📈📉💡🔍⚡🧠🎯]+', '', answer_text)
                clean_answer = re.sub(r'\s+', ' ', clean_answer).strip()
                
                story.append(Paragraph(f"Answer{context_info}:", styles['Normal']))
                story.append(Paragraph(clean_answer, styles['Normal']))
                
                # Key finding
                if session.get('key_finding'):
                    story.append(Paragraph(f"Key Finding: {session['key_finding']}", styles['Italic']))
                
                # Visualization indicator and try to embed actual image
                if session.get('had_visualization'):
                    # Try to find and embed the actual PNG image
                    image_embedded = try_embed_session_image(story, session, analysis_history, dataset_name)
                    if not image_embedded:
                        story.append(Paragraph("📊 Included data visualization (view HTML files for interactive charts)", styles['Italic']))
                
                story.append(Spacer(1, 15))
                
                # Add page break after every 3 user sessions to avoid overcrowding
                if i % 3 == 0 and i < len(user_sessions):
                    story.append(PageBreak())
            
            story.append(Spacer(1, 20))
    
    
    # Recent Visualizations from Files
    story.append(Paragraph("📈 Recent Visualizations", subtitle_style))
    viz_files = get_recent_visualizations(dataset_name)
    if viz_files:
        story.append(Paragraph(f"Found {len(viz_files)} recent visualizations:", styles['Normal']))
        for viz_file in viz_files:
            filename = os.path.basename(viz_file)
            # Extract timestamp from filename if possible
            timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                try:
                    dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    story.append(Paragraph(f"• {filename} (Created: {time_str})", styles['Normal']))
                except:
                    story.append(Paragraph(f"• {filename}", styles['Normal']))
            else:
                story.append(Paragraph(f"• {filename}", styles['Normal']))
    else:
        story.append(Paragraph("No recent visualization files found.", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Statistical Summary
    if dataset_name in state.datasets:
        story.append(Paragraph("📊 Statistical Summary", subtitle_style))
        df = state.datasets[dataset_name]
        numeric_df = df.select_dtypes(include='number')
        
        if len(numeric_df.columns) > 0:
            story.append(Paragraph("Key Statistics for Numeric Features:", styles['Heading3']))
            summary_stats = numeric_df.describe()
            
            # Create summary table
            stats_data = [['Feature', 'Mean', 'Std', 'Min', 'Max']]
            for col in numeric_df.columns[:5]:  # First 5 numeric columns
                stats_data.append([
                    col,
                    f"{summary_stats.loc['mean', col]:.2f}",
                    f"{summary_stats.loc['std', col]:.2f}",
                    f"{summary_stats.loc['min', col]:.2f}",
                    f"{summary_stats.loc['max', col]:.2f}"
                ])
            
            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(stats_table)
    
    story.append(Spacer(1, 30))
    
    # Footer
    story.append(Paragraph("---", styles['Normal']))
    story.append(Paragraph(f"Report generated by AI Data Analysis Agent on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
    
    # Build the PDF
    doc.build(story)
    
    # Get absolute path for return
    abs_path = os.path.abspath(output_path)
    print(f"✅ Enhanced report generated at: {abs_path}")
    
    # Store report path in state for future reference
    if not hasattr(state, 'reports'):
        state.reports = {}  # add dynamically if missing
    state.reports[dataset_name] = abs_path
    
    return abs_path


def add_visualization_to_report(story, viz_html, filename_prefix):
    """Convert Plotly HTML to image and embed in PDF report"""
    try:
        # Method 1: Try to extract Plotly JSON and convert to image
        success = convert_plotly_to_image(story, viz_html, filename_prefix)
        if success:
            return True
        
        # Method 2: Try to use kaleido for static image export
        success = convert_with_kaleido(story, viz_html, filename_prefix)
        if success:
            return True
        
        # Method 3: Fallback - add placeholder with helpful info
        story.append(Paragraph("📊 Visualization Generated", getSampleStyleSheet()['Heading3']))
        story.append(Paragraph("Chart created successfully. View the interactive version in the HTML files saved to visualizations/ folder for full interactive experience.", getSampleStyleSheet()['Italic']))
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not embed visualization: {e}")
        return False

def convert_plotly_to_image(story, viz_html, filename_prefix):
    """Convert Plotly chart to static image using plotly.io"""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from reportlab.platypus import Image
        import re
        import json
        
        # Extract Plotly JSON from HTML
        # Look for the JSON data in the HTML
        json_pattern = r'var\s+gd\s*=\s*document\.getElementById\([^}]+\}\s*,\s*(\{.+?\})\s*,\s*(\{.+?\})'
        match = re.search(json_pattern, viz_html, re.DOTALL)
        
        if not match:
            # Try alternative pattern
            json_pattern = r'Plotly\.newPlot\([^,]+,\s*(\[.+?\])\s*,\s*(\{.+?\})'
            match = re.search(json_pattern, viz_html, re.DOTALL)
        
        if match:
            try:
                # Parse the extracted JSON
                data_str = match.group(1)
                layout_str = match.group(2) if len(match.groups()) > 1 else '{}'
                
                data = json.loads(data_str)
                layout = json.loads(layout_str)
                
                # Create Plotly figure
                fig = go.Figure(data=data, layout=layout)
                
                # Generate image file
                img_path = f"{filename_prefix}.png"
                
                # Try to write image (requires kaleido or orca)
                pio.write_image(fig, img_path, format="png", width=600, height=400, scale=2)
                
                # Add image to PDF
                story.append(Image(img_path, width=400, height=300))
                
                # Clean up image file
                try:
                    os.remove(img_path)
                except:
                    pass
                
                return True
                
            except Exception as e:
                print(f"JSON parsing failed: {e}")
                return False
        
        return False
        
    except ImportError:
        print("Plotly not available for image conversion")
        return False
    except Exception as e:
        print(f"Plotly conversion failed: {e}")
        return False

def convert_with_kaleido(story, viz_html, filename_prefix):
    """Try to use kaleido for image conversion"""
    try:
        # This requires 'pip install kaleido'
        import kaleido
        # Implementation would go here
        # For now, return False to use fallback
        return False
        
    except ImportError:
        # Kaleido not installed
        return False

def try_embed_session_image(story, session, analysis_history, dataset_name):
    """Try to find and embed PNG image for a user analysis session"""
    try:
        from reportlab.platypus import Image
        import glob
        
        # Get session timestamp
        session_timestamp = session.get('timestamp', '')
        if not session_timestamp:
            return False
        
        # Convert timestamp format (from ISO to filename format)
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(session_timestamp.replace('Z', '+00:00'))
            timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        except:
            # Fallback: try to extract timestamp parts
            import re
            match = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})', session_timestamp)
            if match:
                date_part = match.group(1).replace('-', '')
                time_part = match.group(2) + match.group(3) + match.group(4)
                timestamp_str = f"{date_part}_{time_part}"
            else:
                return False
        
        # Look for PNG files with matching timestamp
        viz_dir = "visualizations"
        if os.path.exists(viz_dir):
            pattern = os.path.join(viz_dir, f"{dataset_name}_{timestamp_str}*.png")
            png_files = glob.glob(pattern)
            
            if png_files:
                # Use the first matching PNG file
                png_path = png_files[0]
                if os.path.exists(png_path):
                    story.append(Image(png_path, width=400, height=300))
                    story.append(Paragraph("📊 Chart visualization", getSampleStyleSheet()['Italic']))
                    return True
        
        # Also try to find recent PNG files (within a few minutes of session)
        try:
            session_dt = datetime.fromisoformat(session_timestamp.replace('Z', '+00:00'))
            png_files = glob.glob(os.path.join(viz_dir, f"{dataset_name}_*.png"))
            
            for png_file in png_files:
                try:
                    # Extract timestamp from filename
                    filename = os.path.basename(png_file)
                    timestamp_match = re.search(r'(\d{8})_(\d{6})', filename)
                    if timestamp_match:
                        file_timestamp = f"{timestamp_match.group(1)}_{timestamp_match.group(2)}"
                        file_dt = datetime.strptime(file_timestamp, "%Y%m%d_%H%M%S")
                        
                        # Check if file was created within 2 minutes of session
                        time_diff = abs((session_dt.replace(tzinfo=None) - file_dt).total_seconds())
                        if time_diff <= 120:  # 2 minutes tolerance
                            story.append(Image(png_file, width=400, height=300))
                            story.append(Paragraph("📊 Chart visualization", getSampleStyleSheet()['Italic']))
                            return True
                except:
                    continue
        except:
            pass
        
        return False
        
    except Exception as e:
        print(f"Warning: Could not embed session image: {e}")
        return False


def get_recent_visualizations(dataset_name, max_files=5):
    """Get recent visualization files for the dataset"""
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        return []
    
    pattern = os.path.join(viz_dir, f"{dataset_name}_*.html")
    viz_files = glob.glob(pattern)
    
    # Sort by modification time, most recent first
    viz_files.sort(key=os.path.getmtime, reverse=True)
    
    return viz_files[:max_files]


# Wrapper function to use enhanced or regular report generation
def generate_report(state: STATE, dataset_name: str, output_path="data_insights_report.pdf", enhanced=True, analysis_history=None) -> str:
    """
    Generate a report - enhanced version by default, with fallback to basic version
    """
    if enhanced:
        try:
            # Use enhanced report generation
            enhanced_path = output_path.replace('.pdf', '_enhanced.pdf')
            return generate_enhanced_report(state, dataset_name, enhanced_path, analysis_history)
        except Exception as e:
            print(f"Enhanced report generation failed: {e}")
            print("Falling back to basic report generation...")
    
    # Fallback to basic report generation (original function)
    return generate_basic_report(state, dataset_name, output_path)


def generate_basic_report(state: STATE, dataset_name: str, output_path="data_insights_report.pdf") -> str:
    """
    Basic report generation (original function)
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    import pandas as pd
    import plotly.io as pio
    
    profile = state.profiles.get(dataset_name, {})
    insights = state.insights.get(dataset_name, [])

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"📊 Report for Dataset: {dataset_name}", styles["Title"]))
    story.append(Spacer(1, 24))

    # Profile Summary
    story.append(Paragraph("Profile Summary", styles["Heading2"]))
    if "columns" in profile and isinstance(profile["columns"], list):
        df_profile = pd.DataFrame(profile["columns"])
        table_data = [df_profile.columns.tolist()] + df_profile.values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("⚠️ No profile summary available.", styles["Normal"]))
    story.append(Spacer(1, 24))

    # Insights
    story.append(Paragraph("Insights", styles["Heading2"]))
    if insights:
        for i, ins in enumerate(insights, 1):
            story.append(Paragraph(f"Q{i}: {ins['question']}", styles["Heading3"]))
            story.append(Paragraph(ins["answer"], styles["Normal"]))
            story.append(Spacer(1, 12))

            if ins.get("visualization_html"):
                png_path = f"{dataset_name}_insight_{i}.png"
                try:
                    fig = pio.from_json(ins["visualization_html"])
                    pio.write_image(fig, png_path, format="png", width=600, height=400, scale=2)
                    story.append(Image(png_path, width=400, height=300))
                except Exception:
                    story.append(Paragraph("⚠️ Visualization could not be embedded in PDF.", styles["Normal"]))
            story.append(Spacer(1, 24))
    else:
        story.append(Paragraph("⚠️ No insights available yet.", styles["Normal"]))

    story.append(Spacer(1, 48))
    doc.build(story)

    abs_path = os.path.abspath(output_path)
    print(f"✅ Basic report generated at: {abs_path}")

    # Update global state
    if not hasattr(state, "reports"):
        state.reports = {}  # add dynamically if missing
    state.reports[dataset_name] = abs_path

    return abs_path