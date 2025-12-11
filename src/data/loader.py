import logging
import yaml
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from datasets import load_dataset
from llama_index.core import Document
from src.core.config import CONFIG

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def load_mol_instructions():
    """
    Loads, filters, and caches data from the Mol-Instructions dataset,
    returning a list of LlamaIndex Document objects.
    """
    data_config = CONFIG['data']
    max_samples = data_config['max_samples']
    cache_dir = Path(data_config['cache_dir'])
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    filtered_file = cache_dir / f"cancer_filtered_{max_samples}.json"

    if filtered_file.exists():
        logger.info("Loading cached filtered data...")
        with open(filtered_file, 'r') as f:
            data = yaml.safe_load(f)
    else:
        logger.info("Downloading and filtering Mol-Instructions dataset...")
        try:
            dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions", split="train", streaming=True)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

        cancer_keywords = data_config['keywords']
        data = []
        count = 0
        for example in dataset:
            if count >= max_samples:
                break
            text = f"{example.get('instruction', '')} {example.get('output', '')}".lower()
            if any(k in text for k in cancer_keywords):
                data.append({
                    'instruction': example.get('instruction', ''),
                    'input': example.get('input', ''),
                    'output': example.get('output', ''),
                    'id': count
                })
                count += 1
        
        with open(filtered_file, 'w') as f:
            yaml.dump(data, f, indent=2)
        logger.info(f"Cached {len(data)} filtered samples.")

    logger.info("Converting data to LlamaIndex Documents...")
    documents = [
        Document(
            text=f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}",
            metadata={"source": "Mol-Instructions", "id": item.get('id', 'N/A')}
        ) for item in data
    ]
    logger.info(f"Created {len(documents)} documents.")
    return documents
