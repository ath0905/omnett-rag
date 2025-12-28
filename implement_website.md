This `implement_website.md` guide will walk you through converting your CLI tool into a fully functional, interactive web application using **Streamlit**.

Streamlit is the industry standard for quickly building AI/Data web apps in Python because it requires **zero** HTML, CSS, or JavaScript knowledge.

---

# 🌐 Implementation Guide: OMNeT++ RAG Web Interface

This guide details how to transform your existing command-line RAG tool into a modern Chatbot Website. We will use **Streamlit** to create a user-friendly interface that runs locally in your browser.

## 1. Project Overview

We are moving from a "Loop in Terminal" approach to a "Web Server" approach.

* **Old:** Text input in console → Print text response.
* **New:** Chat bubble UI → Streaming responses → Session History.

We will reuse your existing `chroma_db_local` database, so you **do not** need to rebuild the vector store.

## 2. Updated Prerequisites

You need one additional Python library.

1. Open your `requirements.txt` and add `streamlit` to the end:
```text
langchain
langchain-community
langchain-ollama
langchain-chroma
pypdf
tiktoken
requests
streamlit        <-- NEW

```


2. Install the update:
```bash
pip install -r requirements.txt

```



## 3. Project Folder Structure

We will create a new file `app.py` alongside your existing script.

```text
omnet_rag/
├── chroma_db_local/         # [Existing] Your vector database
├── omnetpp_guide.pdf        # [Existing] The PDF
├── offline_rag.py           # [Old] CLI script
├── app.py                   # [NEW] The Website script
└── requirements.txt         # Updated dependencies

```

## 4. The Website Code (`app.py`)

Create `app.py` and paste the code below.

**Key Changes from CLI:**

* **`@st.cache_resource`**: This decorator tells Streamlit to load the Database and LLM *once* and keep them in memory. Without this, the app would reload the entire database every time you clicked a button.
* **Session State**: We use `st.session_state` to remember the chat history so previous messages don't disappear.

```python
import os
import requests
import time
import streamlit as st

# --- LangChain & Ollama Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
PDF_URL = "https://doc.omnetpp.org/omnetpp/UserGuide.pdf"
LOCAL_PDF_PATH = "omnetpp_guide.pdf"
VECTOR_DB_PATH = "./chroma_db_local"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Models
LLM_MODEL = "mistral"           # Make sure you have this pulled!
EMBED_MODEL = "nomic-embed-text"

# --- Page Config ---
st.set_page_config(
    page_title="OMNeT++ Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- cached functions (Run once) ---

@st.cache_resource
def setup_vector_store():
    """
    Checks for existing DB. If not found, downloads PDF and builds DB.
    Returns the vectorstore object.
    """
    # 1. Download PDF if needed
    if not os.path.exists(LOCAL_PDF_PATH):
        with st.spinner(f"⬇️ Downloading PDF from {PDF_URL}..."):
            response = requests.get(PDF_URL, stream=True)
            with open(LOCAL_PDF_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("✅ Download complete.")

    # 2. Initialize Embeddings
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 3. Load or Build DB
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        # Load existing
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        print("📂 Loaded existing DB")
    else:
        # Build new
        with st.spinner("📚 Parsing PDF and creating database (this takes 1-3 mins)..."):
            loader = PyPDFLoader(LOCAL_PDF_PATH)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=VECTOR_DB_PATH
            )
            st.success("✅ Database created!")
            
    return vectorstore

def get_chain(vectorstore):
    """Creates the RAG chain."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    system_prompt = (
        "You are an expert technical assistant for OMNeT++. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the context contains code (NED, C++, XML), format it properly in markdown blocks. "
        "Do not make up information. If the answer is not in the context, say you don't know."
        "\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# --- Main App Logic ---

st.title("🤖 OMNeT++ RAG Assistant")
st.caption(f"Powered by Local Ollama ({LLM_MODEL}) | Offline Mode")

# 1. Initialize Vector Store (Cached)
try:
    vectorstore = setup_vector_store()
    rag_chain = get_chain(vectorstore)
except Exception as e:
    st.error(f"Failed to setup system: {e}")
    st.stop()

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am ready to answer questions about the OMNeT++ User Guide."}]

# 3. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle User Input
if prompt := st.chat_input("Ask a technical question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        start_time = time.time()
        
        # Run the RAG chain
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                full_response = response['answer']
                
                # Calculate timing
                elapsed = time.time() - start_time
                footer = f"\n\n---\n*Generated in {elapsed:.2f}s using {LLM_MODEL}*"
                
                # Display result
                message_placeholder.markdown(full_response + footer)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")

```

## 5. Execution Plan

### Step 1: Start Ollama

Ensure Ollama is running in the background (check system tray or run `ollama serve`).

### Step 2: Run the Website

Open your terminal in the `omnet_rag` folder and run:

```bash
streamlit run app.py

```

### Step 3: Access the Interface

* Streamlit will automatically open your default web browser to a local URL (usually `http://localhost:8501`).
* Since you already built the database in the previous implementation, the app should load **instantly**.

## 6. Website Features

* **Chat Interface:** Just like ChatGPT, but running on your data.
* **Code Formatting:** The app automatically renders NED, C++, and XML code blocks beautifully.
* **Performance:** Uses caching to ensure the database isn't reloaded on every interaction.
* **Persistence:** The chat history stays visible as long as you don't refresh the browser page.

## 7. Troubleshooting

* **"Model not found"**: Ensure you have pulled the model defined in the `LLM_MODEL` variable (e.g., `ollama pull mistral`).
* **Stuck on "Running..."**: Creating the database for the first time takes CPU power. Check your terminal for progress logs.
* **Port already in use**: If `localhost:8501` is taken, Streamlit will automatically try `8502`. Check the terminal output for the correct URL.
