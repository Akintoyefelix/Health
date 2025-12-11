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

- **Language**: Python 3.11+
- **LLM Engine**: unsloth/Llama-3.2-1B-Instruct (4-bit Quantized via bitsandbytes & accelerate)
- **RAG Framework**: LlamaIndex
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: ChromaDB
- **Data Orchestration**: HuggingFace Datasets, UniProt REST API
- **Interface**: Streamlit
- **Deployment**: Docker, Google Cloud Platform (App Engine / Cloud Run)

## 📦 Installation Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akintoyefelix/Health.git
   cd Health
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t skin-cancer-rag .
   ```

2. **Run the container**
   ```bash
   docker run -p 8080:8080 skin-cancer-rag
   ```

## 📂 System Architecture & API Reference

### 1. Core Logic (`src/core/`)
- **`engine.py`**: Contains the `UniProtEnrichedQueryEngine` class that orchestrates retrieval and generation.
- **`models.py`**: Manages the initialization of the quantized LLM and embedding models.
- **`index.py`**: Handles the creation and loading of the ChromaDB vector index.
- **`config.py`**: Centralized configuration loader.

### 2. Data Management (`src/data/`)
- **`loader.py`**: Responsible for downloading, filtering, and caching the Mol-Instructions dataset.

### 3. Utilities (`src/utils/`)
- **`uniprot.py`**: The bridge to the UniProt API for real-time protein data fetching.

### 4. User Interface (`app.py`)
- **Framework**: **Streamlit**
- **Function**: Provides a chat-based interface for clinicians to interact with the model.
- **Features**: Maintains chat history and renders markdown responses.

### 4. Configuration (`config.yaml`)
- **Role**: Centralized configuration for model names, data paths, and hyperparameters.
- **Benefit**: Allows easy switching of models or datasets without changing code.

## 🚀 Usage Example

```bash
# Start the Streamlit application
streamlit run app.py
```

The application will launch in your browser, allowing you to interactively query the system about skin cancer mutations.

## ☁️ Deployment

### Google Cloud Platform (App Engine)

1. **Deploy to App Engine**
   ```bash
   gcloud app deploy app.yaml
   ```

### Google Cloud Run

1. **Build and Submit**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/skin-cancer-rag
   ```

2. **Deploy**
   ```bash
   gcloud run deploy skin-cancer-rag --image gcr.io/YOUR_PROJECT_ID/skin-cancer-rag --platform managed
   ```

## 📝 Project Summary

**Project**: Skin Cancer Mutation RAG System  
**Domain**: Molecular Science / Oncology  
**Status**: Completed

This project demonstrates a sophisticated application of Applied AI in the biomedical field. By integrating Retrieval-Augmented Generation (RAG) with structured biological APIs (UniProt), the team successfully mitigated the common issue of LLM hallucination. The solution is not merely a theoretical prototype but a production-optimized tool (utilizing quantization) capable of running on accessible hardware. It meets all criteria of the assignment, delivering a specialized, high-utility software solution for molecular science.
