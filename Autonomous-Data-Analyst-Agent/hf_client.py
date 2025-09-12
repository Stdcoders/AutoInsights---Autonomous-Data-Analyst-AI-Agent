# hf_client.py
import os
from huggingface_hub import InferenceClient

def get_hf_client(model_name: str = "mistralai/Mistral-7B-v0.1", api_key: str = None):
    """
    Initialize and return a Hugging Face Inference API client.
    """
    hf_api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
    if not hf_api_key:
        raise ValueError("Hugging Face API key not provided! Set HUGGINGFACE_API_KEY environment variable.")
    
    try:
        client = InferenceClient(token=hf_api_key)
        return client, model_name
    except Exception as e:
        print(f"⚠️ Hugging Face client initialization failed: {e}")
        return None, model_name
