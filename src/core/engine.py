import logging
from typing import List, Dict, Any
from llama_index.core.query_engine import BaseQueryEngine
from src.utils.uniprot import UniProtCache

logger = logging.getLogger(__name__)

class UniProtEnrichedQueryEngine(BaseQueryEngine):
    """
    Custom query engine that enriches context with UniProt data.
    """
    def __init__(self, retriever, llm, prompt_template):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.uniprot_cache = UniProtCache()
        try:
            self.uniprot_cache.preload_cancer_proteins()
        except Exception as e:
            logger.warning(f"Could not preload UniProt data: {e}")
        super().__init__(retriever)

    def _extract_proteins(self, text: str) -> List[str]:
        text_upper = text.upper()
        return [p for p in self.uniprot_cache.cancer_proteins if p in text_upper]

    def _build_context(self, retrieved_nodes: List, protein_info: List[Dict]) -> str:
        context_parts = []
        if retrieved_nodes:
            context_parts.append("### Retrieved Scientific Information:")
            for i, node in enumerate(retrieved_nodes, 1):
                context_parts.append(f"{i}. {node.get_text()[:350]}...")
        
        if any(protein_info):
            context_parts.append("\n### Protein Database Information (from UniProt):")
            for protein in protein_info:
                if protein:
                    context_parts.append(
                        f"- **{protein['gene']}**: {protein['protein_name']}\n"
                        f"  *Function*: {protein['function'][:200]}...")
        return "\n".join(context_parts)

    def _query(self, query_str: str) -> Dict[str, Any]:
        try:
            logger.info("Retrieving documents...")
            retrieved_nodes = self.retriever.retrieve(query_str)
            
            logger.info("Fetching protein information from UniProt...")
            proteins_mentioned = self._extract_proteins(query_str)
            protein_info = [self.uniprot_cache.fetch_protein_info(p) for p in proteins_mentioned]

            logger.info("Building augmented context...")
            context_str = self._build_context(retrieved_nodes, protein_info)
            
            logger.info("Generating answer with LLM...")
            formatted_prompt = self.prompt_template.format(
                context_str=context_str,
                query_str=query_str
            )
            
            response = self.llm.complete(formatted_prompt)
            
            return {"response": str(response), "source_nodes": retrieved_nodes}
        except Exception as e:
            logger.error(f"Error in query execution: {e}")
            raise
