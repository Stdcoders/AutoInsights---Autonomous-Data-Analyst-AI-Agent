from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch

import os

def generate_report(state: "ReportState", dataset_name: str, output_path: str = None) -> str:
    """
    Generate a structured PDF report from the workflow state.
    Includes dataset profile, generated questions, selected insights, and visualizations.
    """
    if not output_path:
        output_path = f"{dataset_name}_report.pdf"

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal = styles['Normal']

    # --- Title Page ---
    story.append(Paragraph(f"📊 Data Report: {dataset_name}", title_style))
    story.append(Spacer(1, 0.4 * inch))

    # --- Dataset Profile ---
    story.append(Paragraph("Dataset Profile", heading_style))
    profile = state.processed_tables.get(dataset_name)
    if profile is not None:
        story.append(Paragraph(f"Rows: {len(profile)} | Columns: {len(profile.columns)}", normal))
        story.append(Spacer(1, 0.2 * inch))

        # Show first 5 rows in table format
        table_data = [list(profile.columns)] + profile.head(5).values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(table)
    story.append(PageBreak())

    # --- Questions ---
    story.append(Paragraph("Generated Analytical Questions", heading_style))
    questions = state.get("generated_questions", {}).get(dataset_name, [])
    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"{i}. {q}", normal))
    story.append(PageBreak())

    # --- Insights ---
    story.append(Paragraph("Insights", heading_style))
    insights = state.get("insights", {}).get(dataset_name, {})
    if insights:
        story.append(Paragraph(insights.get("answer", "No insights generated."), normal))

    # --- Visualization (if available) ---
    if "visualization_path" in insights:
        from reportlab.platypus import Image
        vis_path = insights["visualization_path"]
        if os.path.exists(vis_path):
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Visualization:", heading_style))
            story.append(Image(vis_path, width=5*inch, height=3*inch))

    # Build PDF
    doc.build(story)

    # Save path in state
    state.pdf_path = output_path
    return output_path