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
# MODIFIED: Accepts api_key explicitly
def get_llm(api_key):
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=api_key # Pass the key directly here
    )

# 2. Embeddings (Running locally on CPU)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# 3. Process Document & Create Vector Store
def process_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return vectorstore

# 4. Create the RAG Chain
# MODIFIED: Accepts groq_api_key explicitly
def create_rag_chain(vectorstore, groq_api_key):
    llm = get_llm(groq_api_key) # Pass the key to the LLM
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
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
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain