import google.generativeai as genai
import os
from dotenv import load_dotenv

# ================== Configure Gemini ==================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment. Please set it in your .env file.")

genai.configure(api_key=API_KEY)


def analyze_documents_with_llm(documents, data_type, model: str = "gemini-1.5-flash"):
    """
    Analyze unstructured documents (text/pdf/image OCR output) using Gemini.
    """
    if isinstance(documents, list):
        text_content = "\n".join(
            [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in documents]
        )
    else:
        text_content = str(documents)

    prompt = f"""
    You are an expert analyst. Analyze the following {data_type} content.
    Provide:
    - Key insights
    - Possible structured information
    - Issues or anomalies
    -----
    {text_content[:3000]}  # truncated if too long
    """

    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config={"temperature": 0.4, "max_output_tokens": 800}
        )
        summary = response.candidates[0].content.parts[0].text
    except Exception as e:
        summary = f"⚠️ Gemini analysis failed: {e}"

    return {"llm_summary": summary}
