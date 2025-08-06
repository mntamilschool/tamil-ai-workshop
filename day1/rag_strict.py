# pip install langchain langchain-community langchain-ollama chromadb pypdf

import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # Import PromptTemplate

# --- Configuration ---
# Directory where your documents are stored
DATA_DIR = "data"
# The name of the Ollama model to use
OLLAMA_MODEL = "gemma" 
# The directory for ChromaDB persistence (if you want to save the vector store)
# For this simple example, we'll keep it in-memory, so this path isn't strictly needed
# but useful to know for future expansion.
CHROMA_DB_DIR = "chroma_db"

# --- Custom Prompt Template ---
# This template guides the LLM to answer based on the context, but also allows it to use
# its general knowledge if the answer is not found in the provided context.
CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of context to answer the user's question.
If you don't know the answer based on the provided context, just say that you don't know,
or use your general knowledge to answer if the question is general.
Do not make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""

# --- 1. Load Documents ---
def load_documents(data_directory):
    """Loads documents from the specified directory."""
    documents = []
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        if filename.endswith(".txt"):
            print(f"Loading text file: {filename}")
            loader = TextLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            print(f"Loading PDF file: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        else:
            print(f"Skipping unsupported file: {filename}")
    return documents

# --- 2. Split Documents into Chunks ---
def split_documents(documents):
    """Splits documents into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Size of each text chunk
        chunk_overlap=200, # Overlap between chunks to maintain context
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# --- 3. Create Vector Store (Knowledge Base) ---
def create_vector_store(chunks):
    """Creates an in-memory Chroma vector store from document chunks."""
    print("Creating embeddings and building vector store...")
    # Initialize Ollama embeddings for converting text to numerical vectors
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # Create ChromaDB from the document chunks and their embeddings
    # This will create an in-memory vector store for this run
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        # persist_directory=CHROMA_DB_DIR # Uncomment to persist the DB to disk
    )
    print("Vector store created.")
    return vector_store

# --- 4. Initialize LLM and RAG Chain ---
def setup_rag_chain(vector_store):
    """Sets up the RAG chain using Ollama LLM and the vector store."""
    print(f"Initializing Ollama LLM with model: {OLLAMA_MODEL}")
    # Initialize the Ollama language model (Gemma 2B)
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # Create a PromptTemplate object
    qa_prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Create a RetrievalQA chain
    # This chain will:
    # 1. Take a question.
    # 2. Use the vector store to retrieve relevant document chunks.
    # 3. Pass the question and the retrieved chunks to the LLM for generation.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" means it stuffs all retrieved documents into one prompt
        retriever=vector_store.as_retriever(),
        return_source_documents=True, # Optionally return the source documents that were used
        chain_type_kwargs={"prompt": qa_prompt} # Pass the custom prompt here
    )
    print("RAG chain setup complete.")
    return qa_chain

# --- Main Execution ---
def main():
    print("--- Simple RAG Project with Ollama and Gemma ---")

    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and add your documents.")
        return

    # 1. Load documents
    documents = load_documents(DATA_DIR)
    if not documents:
        print("No documents found in the data directory. Please add some text files or PDFs.")
        return

    # 2. Split documents
    chunks = split_documents(documents)

    # 3. Create vector store
    vector_store = create_vector_store(chunks)

    # 4. Setup RAG chain
    qa_chain = setup_rag_chain(vector_store)

    print("\nReady to answer questions! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            print("Exiting RAG application. Goodbye!")
            break

        try:
            # Invoke the RAG chain with the user's query
            result = qa_chain.invoke({"query": query})
            print("\n--- Answer ---")
            print(result["result"])
            if result.get("source_documents"):
                print("\n--- Sources Used ---")
                for i, doc in enumerate(result["source_documents"]):
                    print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown Source')}")
                    # print(f"Content snippet: {doc.page_content[:150]}...") # Uncomment to see snippets
            print("--------------")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure Ollama is running and the 'gemma' model is downloaded.")

if __name__ == "__main__":
    main()
