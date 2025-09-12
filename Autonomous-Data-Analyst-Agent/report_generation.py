import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import plotly.io as pio
from typing import List, Dict, Any
from hf_client import get_hf_client

class ProactiveReportGeneratorPDF:
    def __init__(self):
        self.client, self.model_name = get_hf_client()

    def _call_hf_model(self, prompt: str) -> str:
        """Call Hugging Face model for report generation"""
        if not self.client:
            return ""
        try:
            output = self.client.text_generation(
                model=self.model_name,
                inputs=prompt,
                max_new_tokens=400
            )
            if isinstance(output, list) and len(output) > 0:
                return output[0].get("generated_text", "")
            return ""
        except Exception as e:
            print(f"⚠️ Hugging Face model call failed: {e}")
            return ""

    def _plot_to_image(self, html_plot: str) -> io.BytesIO:
        """Convert Plotly HTML to PNG image"""
        fig = pio.from_html(html_plot)
        buf = io.BytesIO()
        fig.write_image(buf, format='PNG')
        buf.seek(0)
        return buf

    def generate_pdf_report(self, insights: List[Dict[str, Any]], filename: str = "Proactive_Report.pdf", title: str = "Proactive Data Analysis Report") -> str:
        """Generate a PDF report with title, summary, and visualizations"""
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph(f"<b>{title}</b>", styles['Title']))
        elements.append(Spacer(1, 20))

        # Executive Summary
        summary_prompt = "Summarize the following insights professionally:\n" + "\n".join([f"{i+1}. {insight['answer']}" for i, insight in enumerate(insights)])
        summary_text = self._call_hf_model(summary_prompt)
        elements.append(Paragraph("<b>Executive Summary:</b>", styles['Heading2']))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(summary_text, styles['BodyText']))
        elements.append(Spacer(1, 20))

        # Insights
        elements.append(Paragraph("<b>Insights:</b>", styles['Heading2']))
        elements.append(Spacer(1, 10))
        for idx, insight in enumerate(insights, start=1):
            elements.append(Paragraph(f"<b>Insight {idx}:</b> {insight['answer']}", styles['BodyText']))
            elements.append(Spacer(1, 5))
            viz_html = insight.get("visualization_html")
            if viz_html:
                try:
                    img_buf = self._plot_to_image(viz_html)
                    elements.append(Image(img_buf, width=450, height=300))
                    elements.append(Spacer(1, 15))
                except Exception as e:
                    elements.append(Paragraph(f"⚠️ Visualization failed: {e}", styles['BodyText']))
                    elements.append(Spacer(1, 10))

        doc.build(elements)
        print(f"✅ PDF report generated: {filename}")
        return filename

    def generate_report(self, insights: List[Dict[str, Any]], filename: str = "Proactive_Report.pdf") -> str:
        return self.generate_pdf_report(insights, filename)

    # ----------------- Run method for LangGraph -----------------
    def run(self, insights: List[Dict[str, Any]], filename: str = "Proactive_Report.pdf") -> str:
        """
        Entry point for LangGraph workflow.
        Takes a list of insights and optional filename.
        Returns the generated PDF file path.
        """
        if not insights:
            print("⚠️ No insights provided to generate PDF report.")
            return ""
        return self.generate_report(insights, filename)
