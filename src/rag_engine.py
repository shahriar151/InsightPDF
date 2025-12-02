import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the LLM (Llama 3.3 via Groq)
def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3, # Low temp for factual answers
    )

# 2. Embeddings (Running locally on CPU)
#This embeddings model is better(BAAI/bge-small-en-v1.5)
# src/rag_engine.py

# 2. Embeddings (Running locally on CPU)
def get_embeddings():
    # Ensuring the model is correctly referenced
    # We explicitly use the model name that is confirmed to work with HuggingFaceEmbeddings
    # If this still fails, the error is likely due to memory limits when loading the model.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # The 'model_kwargs' is optional but ensures it's set up for CPU usage
        model_kwargs={'device': 'cpu'}
    )
# 3. Process Document & Create Vector Store
# 3. Process Document & Create Vector Store
def process_document(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # Create Vector Store (Using ChromaDB)
    embeddings = get_embeddings()
    # The `persist_directory` is optional but allows Chroma to save the embeddings
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"  # <-- CHROMA CREATES THIS FOLDER
    )
    return vectorstore

# 4. Create the RAG Chain
def create_rag_chain(vectorstore):
    llm = get_llm()
    
    # Retrieve top 3 relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # System Prompt: This is where you tell it to act like a pro
    system_prompt = (
        "You are an expert academic assistant. Use the provided context to answer the question. "
        "If the answer is not in the context, say you don't know. "
        "Always cite the page number if available in the context metadata.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Standard LangChain RAG pipeline
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain