from langchain.llms.base import LLM
from typing import Optional, Any, Mapping, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

class GemmaLLM(LLM):
    model_name: str = "google/flan-t5-base"  # Pydantic field
    hf_pipeline: Optional[Any] = None        # store Hugging Face pipeline

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # crucial: call LLM's init

        # Initialize Hugging Face tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if "gemma" in self.model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
                offload_folder="offload_dir"
            )
            task = "text-generation"
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            task = "text2text-generation"

        self.hf_pipeline = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if "gemma" in self.model_name.lower() else -1
        )

    @property
    def _llm_type(self) -> str:
        return "gemma"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        outputs = self.hf_pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            #return_full_text=False
        )
        return outputs[0]["generated_text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}
