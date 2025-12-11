import logging
from llama_index.core import Settings, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from src.core.config import CONFIG
from src.core.models import configure_models
from src.data.loader import load_mol_instructions
from src.core.index import get_or_build_index
from src.core.engine import UniProtEnrichedQueryEngine

logger = logging.getLogger(__name__)

def create_rag_engine():
    """
    Main function to set up and return the complete RAG pipeline.
    """
    logger.info("="*50)
    logger.info("INITIALIZING RAG PIPELINE")
    
    try:
        configure_models()
        documents = load_mol_instructions()
        if not documents:
            raise ValueError("No documents loaded from dataset")
        
        index = get_or_build_index(documents)
        
        retriever_config = CONFIG['retriever']
        retriever = VectorIndexRetriever(index=index, similarity_top_k=retriever_config['similarity_top_k'])

        prompt_template = PromptTemplate(CONFIG['prompt_template'])

        logger.info("Creating UniProt-enriched query engine...")
        query_engine = UniProtEnrichedQueryEngine(
            retriever=retriever,
            llm=Settings.llm,
            prompt_template=prompt_template
        )
        
        logger.info("✅ RAG Pipeline Initialized Successfully!")
        logger.info("="*50)
        
        return query_engine
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
