import streamlit as st
import os
import sys
import tempfile

# 1. --- PAGE CONFIG ---
st.set_page_config(page_title="InsightPDF", layout="wide")
st.title("📄 InsightPDF: Private Document Assistant")

# 2. --- SECRETS SETUP ---
# Grab the API Key directly from secrets
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY not found in Streamlit Secrets!")
    st.stop()

# 3. --- RAG ENGINE IMPORT ---
import sys
# --- FIX FOR STREAMLIT DEPLOYMENT PATH ---
# This adds the project root to the system path, allowing imports from 'src'.
# The project root on Streamlit Cloud is '/mount/src/insightpdf'.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# 2. --- RAG ENGINE IMPORT (Must come AFTER env setup) ---
# Now we can just use the absolute import path 'src.rag_engine'
# or since we added 'src' to the path, we can try to import directly.
try:
    from rag_engine import process_document, create_rag_chain
except ModuleNotFoundError:
    # If the app is run from a location where 'src' is not the root, 
    # the explicit 'src.' prefix might be needed, but the sys.path fix 
    # usually allows the direct name to work if the file is in 'src'.
    # Let's use the explicit 'src.rag_engine' import to be safe if 'src'
    # is the actual package name.

    # We will prioritize the safer explicit import since we modified the path.
    # If the first import fails, it means the structure is slightly different 
    # than assumed, so we try the other way.
    
    # Try the explicit import from 'src' as the package name:
    try:
        from src.rag_engine import process_document, create_rag_chain
    except ImportError as e:
        st.error(f"Failed to import rag_engine. Please check file structure. Error: {e}")
        st.stop()

# --- PASSWORD PROTECTION ---
def check_password():
    if "APP_PASSWORD" not in st.secrets:
        # If running locally without secrets.toml, you might want to bypass or warn
        # But for Cloud deployment, this ensures security
        return True

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.sidebar.header("Login")
        password_input = st.sidebar.text_input("Enter App Password", type="password")
        
        if password_input == st.secrets["APP_PASSWORD"]:
            st.session_state.password_correct = True
            st.rerun() 
        elif password_input:
            st.sidebar.error("Incorrect Password")
            
    return st.session_state.password_correct

if not check_password():
    st.stop() 

st.markdown("Powered by **Llama-3.3-70B** & **LangChain**")

# Sidebar for File Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner("Analyzing document... (Computing Embeddings)"):
            if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = process_document(tmp_path)
                # MODIFIED: Passing the API key here
                st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore, groq_api_key)
                st.success("Document Processed!")
            
            os.remove(tmp_path)

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "rag_chain" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({"input": prompt})
                answer = response['answer']
                sources = response['context']
                unique_sources = list(set([f"Page {doc.metadata.get('page', 0) + 1}" for doc in sources]))
                
                full_response = f"{answer}\n\n**Sources:** {', '.join(unique_sources)}"
                st.markdown(full_response)
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Please upload a PDF first.")