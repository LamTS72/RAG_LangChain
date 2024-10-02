import torch
import sys
from transformers import(
        BitsAndBytesConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModel,
        pipeline,
)
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
# from utils import ConfigKey
# config = ConfigKey()

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
)

def get_hf_model(model_name="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=1024, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                low_cpu_less_usage=True,
                device_map="mps"
        )
        return model

def get_hf_model_gguf(model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", max_new_tokens=1024, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file="mistral-7b-instruct-v0.2.Q4_K_S.gguf")
        print(model)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        model_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                pad_token_id = tokenizer.eos_token_id,
                device_map="auto"
        )
        llm = HuggingFacePipeline(
                pipeline=model_pipeline,
                model_kwargs={
                        "temperature":0.2
                }
        )
        return llm
# get_hf_model_gguf()