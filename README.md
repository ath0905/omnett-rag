```markdown
#  Offline RAG for OMNeT++ User Guide

A completely local, privacy-focused **Retrieval-Augmented Generation (RAG)** system designed to answer technical questions about the [OMNeT++ Simulation Manual](https://doc.omnetpp.org/omnetpp/UserGuide.pdf).

It runs 100% offline using **Ollama** and **Python**, featuring both a **modern Web Interface** and a standard **Command Line** tool. No API keys, no cloud costs, and zero data leakage.

##  Features
* **Zero Cost:** Uses open-source local LLMs (Mistral/Llama/Qwen).
* **Privacy First:** Your queries and documents never leave your machine.
* **Dual Interfaces:** Choose between a Chatbot Website (Streamlit) or a Terminal Tool.
* **Persistent Memory:** Vectors are saved locally (`./chroma_db_local`), so you only need to ingest the PDF once.
* **Smart Retrieval:** Uses semantic search to find relevant context, even if keywords don't match exactly.

---

##  Prerequisites

### 1. Install Ollama
You need the Ollama backend to run the AI models.
1.  Download from [ollama.com](https://ollama.com).
2.  Install and ensure it is running (`ollama serve`).

### 2. Download Models
Open your terminal and pull the required models. We use **Mistral** for logic and **Nomic** for embeddings.

```bash
# The Brain (LLM)
ollama pull mistral

# The Embedder (Vector Converter)
ollama pull nomic-embed-text

```

### 3. Python Requirements

* Python 3.9 or higher.
* Install dependencies:

```bash
pip install -r requirements.txt

```

*(Ensure your `requirements.txt` contains: `streamlit`, `langchain`, `langchain-community`, `langchain-ollama`, `langchain-chroma`, `pypdf`, `tiktoken`, `requests`)*

---

##  Usage

You can run the assistant in two ways. Both use the same database, so you don't need to re-ingest data when switching.

### Option 1: Web Interface (Recommended)

This launches a ChatGPT-style interface in your browser.

```bash
streamlit run app.py

```

* **First Run:** It will automatically download the PDF and build the database (taking 1-3 minutes).
* **Access:** It typically opens at `http://localhost:8501`.

### Option 2: Command Line Interface

This runs the assistant directly in your terminal.

```bash
python offline_rag.py

```

---

##  How It Works (The Process)

This system follows a standard RAG pipeline, divided into two main phases:

### Phase 1: Ingestion (Building the Brain)

*Occurs only on the first run.*

1. **Document Loading:** The script downloads `UserGuide.pdf` and uses `PyPDFLoader` to extract raw text.
2. **Chunking:** The text is split into smaller, manageable pieces (1000 characters) using `RecursiveCharacterTextSplitter`. This ensures that code blocks and paragraphs are kept intact.
3. **Embedding:** Each chunk is passed through the local `nomic-embed-text` model. This converts text into a list of numbers (vectors) that represent the *meaning* of the text.
4. **Storage:** These vectors are saved into a local **ChromaDB** folder (`chroma_db_local`).

### Phase 2: Retrieval (Answering Questions)

*Occurs every time you ask a question.*

1. **Query Embedding:** When you ask a question, it is converted into numbers using the same embedding model.
2. **Similarity Search:** ChromaDB finds the top 5 chunks of text from the PDF that are mathematically most similar to your question.
3. **Augmentation:** These 5 chunks are pasted into a system prompt (hidden from you) that looks like: *"Here is some context from the manual... answer the user's question using this context."*
4. **Generation:** The **Mistral** model reads the context and generates your answer based *only* on the facts in the manual.

---

##  Configuration

You can customize the system by editing variables at the top of `app.py` or `offline_rag.py`:

* **Change the Model:**
```python
LLM_MODEL = "qwen2.5"  # Switch to Qwen (requires 'ollama pull qwen2.5')

```


* **Change the Document:**
Update `PDF_URL` to point to any other PDF documentation you wish to query.

---

##  Troubleshooting

* **Error: `model "mistral" not found**`: You forgot to run `ollama pull mistral`.
* **Error: `ConnectionRefused**`: Ollama is not running. Start the application.
* **The AI is hallucinating code:** Small models sometimes struggle with complex C++ syntax. Ensure you are using `mistral` or `qwen2.5` rather than smaller models like `llama3.2`.
* **Streamlit Port Error:** If `localhost:8501` is busy, Streamlit will try port `8502`. Check your terminal output for the correct URL.

```

```
