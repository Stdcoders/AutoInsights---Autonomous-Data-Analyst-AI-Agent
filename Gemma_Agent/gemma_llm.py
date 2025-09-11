# ---------------- gemma_llm.py ----------------
from langchain.llms.base import LLM
from typing import Optional, List, Any, Mapping
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# Add at the top of gemma_llm.py
from huggingface_hub import login
import os

# Login to Hugging Face
def hf_login():
    token = os.getenv("HF_API_TOKEN")
    if not token:
        print("❌ HF_API_TOKEN not found in environment variables.")
        print("Please set your Hugging Face token:")
        print("1. Get your token from: https://huggingface.co/settings/tokens")
        print("2. Set it as environment variable: export HF_API_TOKEN=your_token_here")
        print("Or enter it manually when prompted.")
        token = input("Enter your Hugging Face token: ")
    login(token=token)

class GemmaLLM(LLM):
    """
    A custom LLM class for Gemma models using Hugging Face Transformers.
    """
    model_name: str = "google/gemma-2b-it"  # or "google/gemma-2b-it"
    pipeline: Any = None

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if model_name:
            self.model_name = model_name
    
    # Add login before loading the model
        hf_login()
    
        print(f"Loading Gemma model: {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
        # Create a text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

    @property
    def _llm_type(self) -> str:
        return "gemma"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Generate text using the pipeline
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt_formatted = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(
            prompt_formatted,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            return_full_text=False  # Do not include the input prompt in the output
        )
        return outputs[0]['generated_text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}