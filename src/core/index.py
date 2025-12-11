import logging
from pathlib import Path
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.core.config import CONFIG

logger = logging.getLogger(__name__)

def get_or_build_index(documents):
    """
    Builds or loads a persistent ChromaDB vector index.
    """
    vs_config = CONFIG['vector_store']
    parser_config = CONFIG.get('node_parser', {'chunk_size': 512})
    db_dir = Path(vs_config['db_directory'])
    db_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Setting up ChromaDB in {db_dir}...")
    db = chromadb.PersistentClient(path=str(db_dir))
    chroma_collection = db.get_or_create_collection(vs_config['collection_name'])
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    if chroma_collection.count() == 0:
        logger.info("Building new vector index...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[SentenceSplitter(chunk_size=parser_config['chunk_size'])]
        )
        logger.info("Vector index built and persisted.")
    else:
        logger.info("Loading existing vector index from ChromaDB.")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            transformations=[SentenceSplitter(chunk_size=parser_config['chunk_size'])]
        )
    return index
