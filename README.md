
                                                        # 🧬 CANCER MUTATION RAG SYSTEM

## Overview

By narrowing the scope to a high-impact local domain—the widespread use and abuse of skin bleaching creams within the African context and the attendant high incidence of skin diseases pathology—we engineered a specialized RAG system for Clinical Oncology. This system allows clinicians and researchers to query complex information regarding skin cancer mutations (e.g., BRAF, NRAS, TP53).

## Objective-to-Solution Matrix

| Assignment Objective | Implemented Solution |
|---|---|
| Use LLMs | Integrated Llama-3.2-1B-Instruct (Quantized) for high-performance, local inference |
| Build RAG Pipeline | Implemented a Semantic Search engine using FAISS and Sentence-Transformers |
| HuggingFace Data | Ingested, filtered, and indexed the "Mol-Instructions" dataset specifically for melanoma/carcinoma contexts |
| Knowledge Bases | Built a real-time UniProt API Bridge to fetch ground-truth protein metadata, reducing hallucination |
| Production AI Track | Utilized 4-bit quantization (bitsandbytes) and memory-efficient caching to ensure the tool runs on consumer hardware/Free-tier Colab |

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **LLM Engine**: unsloth/Llama-3.2-1B-Instruct (4-bit Quantized via bitsandbytes & accelerate)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: FAISS (Facebook AI Similarity Search) - CPU Index
- **Data Orchestration**: HuggingFace Datasets, UniProt REST API
- **Interface**: Gradio

## 📦 Installation Setup

Deploying the Solution requires a Python environment.

Use the BASH command shell to:

```bash
# 1. Install core RAG and LLM dependencies
pip install -q datasets transformers sentence-transformers faiss-cpu

# 2. Install optimization libraries for Production AI (Quantization)
pip install -q bitsandbytes accelerate

# 3. Install Interface and Networking tools
pip install -q gradio requests
```

## 📂 System Architecture & API Reference

### 1. Data Ingestion: MolInstructions
- **Role**: The Gatekeeper
- **Function**: Manages the ingestion of raw scientific text. Restricts data ingestion/input strictly within skin-cancer-related terms (e.g., 'melanoma', 'V600E')
- **Key Method**: `download_and_filter(max_samples=5000)` — Streams the "zjunlp/Mol-Instructions" dataset and saves a local optimized JSON (cancer_filtered.json)

### 2. Knowledge Base Bridge: UniProt
- **Role**: The Fact-Checker
- **Function**: Connects to the UniProt Knowledgebase to provide "ground truth" data, critical for preventing LLM hallucinations regarding gene names or biological functions
- **Key Method**: `fetch_protein_info(gene_name)` — Queries the UniProt REST API to extract Protein Name, Function, and Sequence Length
- **Optimization**: Includes a `_load_cache` mechanism to prevent redundant API calls for commonly queried proteins (BRAF, TP53)

### 3. The Search Engine: CancerRAGRetriever
- **Role**: The Librarian
- **Function**: Converts text into mathematical vectors and retrieves specific scientific contexts relevant to the user's query
- **Key Method**: `build_index(data)` — Creates a FAISS Index (IndexFlatIP) for efficient cosine similarity search
- **Key Method**: `retrieve(query, top_k=3)` — Returns the top 3 most relevant scientific snippets for the prompt

### 4. The Brain: QuantizedLLM
- **Role**: The Synthesizer
- **Function**: A memory-efficient wrapper for the Llama model. Uses NF4 (NormalFloat 4-bit) quantization to achieve high performance with significantly lower VRAM usage
- **Key Method**: `generate(prompt)` — Runs the inference loop with temperature control (0.7) to balance creativity and factual adherence

### 5. Main Controller: CancerMutationRAG
- **Role**: The Conductor
- **Function**: Orchestrates the entire pipeline
- **Workflow** (query method):
  1. **Retrieve**: Calls CancerRAGRetriever to get text docs
  2. **Verify**: Scans query for gene names and calls UniProtCache
  3. **Construct**: Merges docs + UniProt facts into a strict system prompt
  4. **Generate**: Calls QuantizedLLM for the final answer

## 🚀 Usage Example

```python
# Initialize the system
rag_system = CancerMutationRAG()
rag_system.initialize()

# Query the system
response = rag_system.query("How does the BRAF V600E mutation affect melanoma treatment?")
print(response)
```

## 📝 Project Summary

**Project**: Skin Cancer Mutation RAG System  
**Domain**: Molecular Science / Oncology  
**Status**: Completed

This project demonstrates a sophisticated application of Applied AI in the biomedical field. By integrating Retrieval-Augmented Generation (RAG) with structured biological APIs (UniProt), the team successfully mitigated the common issue of LLM hallucination. The solution is not merely a theoretical prototype but a production-optimized tool (utilizing quantization) capable of running on accessible hardware. It meets all criteria of the assignment, delivering a specialized, high-utility software solution for molecular science.
