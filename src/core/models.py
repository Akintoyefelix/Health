import logging
import torch
from transformers import BitsAndBytesConfig
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.core.config import CONFIG

logger = logging.getLogger(__name__)

def configure_models():
    """Initializes and configures the global LLM and embedding models."""
    logger.info("Configuring models...")
    
    model_config = CONFIG['models']
    llm_gen_config = CONFIG.get('llm_generation', {
        'context_window': 2048,
        'max_new_tokens': 256,
        'temperature': 0.7,
        'do_sample': True
    })

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    Settings.llm = HuggingFaceLLM(
        model_name=model_config['llm'],
        tokenizer_name=model_config['llm'],
        context_window=llm_gen_config['context_window'],
        max_new_tokens=llm_gen_config['max_new_tokens'],
        model_kwargs={"quantization_config": quantization_config},
        generate_kwargs={
            "temperature": llm_gen_config['temperature'],
            "do_sample": llm_gen_config['do_sample']
        },
        device_map="auto",
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name=model_config['embedding'])
    logger.info("Models configured successfully.")
