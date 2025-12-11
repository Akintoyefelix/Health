"""
Streamlit Web Application for the Skin Cancer Mutation AI Assistant.
"""

import streamlit as st
import logging
from src.main import create_rag_engine

logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Skin Cancer AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #f4f6f8;
    }
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=80)
    st.title("Onco-RAG Assistant")
    st.markdown("---")
    st.markdown("""
    **System Status:**
    - 🟢 LLM Engine: **Llama-3.2-1B**
    - 🟢 Knowledge Base: **Mol-Instructions**
    - 🟢 Fact-Checking: **UniProt API**
    """)
    st.markdown("---")
    st.info("This tool is designed for clinical research and educational purposes.")

# --- Main Interface ---
st.title("🧬 Skin Cancer Mutation Analysis System")
st.markdown("""
Welcome to the **RAG-powered Clinical Assistant**. 
Ask questions about skin cancer mutations (e.g., *BRAF V600E*, *NRAS*, *TP53*) to get evidence-based answers.
""")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    with st.spinner("Initializing AI Engine... (This may take a minute)"):
        try:
            st.session_state.rag_engine = create_rag_engine()
            st.success("System Ready!")
        except Exception as e:
            st.error(f"System Initialization Failed: {e}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about a mutation (e.g., 'What is the clinical significance of BRAF V600E?')..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing scientific literature and protein databases..."):
            try:
                response_obj = st.session_state.rag_engine.query(prompt)
                response_text = response_obj["response"]
                
                st.markdown(response_text)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Show sources in an expander
                with st.expander("📚 View Scientific Sources"):
                    for node in response_obj["source_nodes"]:
                        st.markdown(f"- {node.get_text()[:200]}...")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")


# --- Initialization and Caching ---
@st.cache_resource
def get_rag_engine():
    """Initializes and returns the RAG query engine."""
    with st.spinner("Initializing AI System... This may take a moment on first launch."):
        try:
            engine = create_rag_engine()
            return engine
        except Exception as e:
            st.error(f"Fatal Error: Could not initialize the RAG engine. Details: {str(e)}")
            logger.error(f"RAG engine initialization failed: {e}")
            st.stop()

try:
    query_engine = get_rag_engine()
except Exception as e:
    st.error("Application failed to start. Please check the logs.")
    st.stop()


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dna-helix.png", width=70)
    st.title("AI Assistant")
    st.markdown("---")
    st.markdown("""
    **About this App:**
    This is a Retrieval-Augmented Generation (RAG) system designed to answer questions about skin cancer mutations.

    **Technology Stack:**
    - **LLM:** `Llama-3.2-1B (Quantized)`
    - **Embeddings:** `all-MiniLM-L6-v2`
    - **Vector Store:** `ChromaDB`
    - **Framework:** `LlamaIndex`
    """)
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")
        st.rerun()

# --- Main Application ---
st.title("🧬 Skin Cancer Mutation AI Assistant")
st.markdown("Ask a question about a skin cancer mutation, its effects, or related therapies.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("e.g., 'What is the clinical significance of BRAF V600E?'"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Searching scientific literature and databases..."):
            try:
                # Query the RAG engine
                response_data = query_engine.query(prompt)
                
                # Extract the answer and display it
                answer = response_data.get('response', 'No response generated.')
                st.markdown(answer)

                # Display source documents in an expander for traceability
                with st.expander("📚 View Sources"):
                    source_nodes = response_data.get('source_nodes', [])
                    if source_nodes:
                        for node in source_nodes:
                            st.markdown(f"**Score:** {getattr(node, 'score', 'N/A'):.3f}")
                            metadata = getattr(node, 'metadata', {})
                            st.caption(f"**Source:** `{metadata.get('source', 'N/A')}`")
                            text = node.get_text() if hasattr(node, 'get_text') else str(node)
                            st.info(text[:400] + "...")
                            st.markdown("---")
                    else:
                        st.warning("No specific source documents were retrieved for this query.")
                
                # Add the full response to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_message = f"Sorry, an error occurred while processing your request. Please try again."
                st.error(error_message)
                logger.error(f"Query processing error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": error_message})