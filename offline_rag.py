import os
import requests
import time

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

# Text Splitting Settings
CHUNK_SIZE = 1000       # Characters per chunk
CHUNK_OVERLAP = 200     # Overlap to maintain context

# Ollama Models
LLM_MODEL = "mistral"          # Options: "mistral", "llama3.1", "qwen2.5"
EMBED_MODEL = "nomic-embed-text"

def download_pdf(url, save_path):
    """Downloads the PDF if it doesn't exist locally."""
    if os.path.exists(save_path):
        return
    
    print(f"⬇️  Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Download complete.")
    except Exception as e:
        raise Exception(f"Failed to download PDF: {e}")

def build_vector_store():
    """Ingests PDF, splits text, creates local embeddings, and saves to ChromaDB."""
    print("📚 Loading PDF (this may take a moment)...")
    loader = PyPDFLoader(LOCAL_PDF_PATH)
    docs = loader.load()
    
    print(f"✂️  Splitting {len(docs)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"   Created {len(splits)} chunks.")

    print(f"💾 Generating embeddings with '{EMBED_MODEL}'...")
    print("   (This runs on your CPU and might take 1-3 minutes first time)")
    
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print("✅ Vector store created and saved.")
    return vectorstore

def get_rag_chain(vectorstore):
    """Creates the retrieval chain using the local LLM."""
    
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.1,  # Low temperature for factual accuracy
    )
    
    # Retrieve top 5 most relevant chunks
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

def main():
    # 1. Setup: Ensure PDF exists
    download_pdf(PDF_URL, LOCAL_PDF_PATH)

    # 2. Database: Load existing or build new
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        print("📂 Loading existing Local Vector Store...")
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    else:
        print("⚠️ No existing database found. Initializing build process...")
        vectorstore = build_vector_store()

    # 3. Chain: Connect LLM to Database
    rag_chain = get_rag_chain(vectorstore)

    # 4. Loop: specific user interaction
    print(f"\n🤖 OMNeT++ Assistant ({LLM_MODEL}) Ready!")
    print("   Type 'exit' to quit.\n")
    print("-" * 60)
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
            
        print("Thinking...", end="", flush=True)
        start_time = time.time()
        
        try:
            response = rag_chain.invoke({"input": query})
            elapsed = time.time() - start_time
            
            print(f"\rAI ({elapsed:.1f}s): \n{response['answer']}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            
        print("-" * 60)

if __name__ == "__main__":
    main()
