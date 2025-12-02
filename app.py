import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
# We wrap the import in try-except to handle local vs cloud paths comfortably
try:
    from src.rag_engine import process_document, create_rag_chain
except ImportError:
    # This helps if python doesn't recognize the src path immediately
    from rag_engine import process_document, create_rag_chain

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="InsightPDF", layout="wide")

st.title("📄 InsightPDF: Private Document Assistant")

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Check if we are running locally (no secrets) or on cloud
    # If strictly local development, we can skip password or hardcode a dev one
    if "APP_PASSWORD" not in st.secrets:
        # If no secret is set (like locally without .streamlit/secrets.toml), let it pass
        # OR better: Warn the user. For this example, we proceed if no secret is found locally.
        return True

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.sidebar.header("Login")
        password_input = st.sidebar.text_input("Enter App Password", type="password")
        
        if password_input == st.secrets["APP_PASSWORD"]:
            st.session_state.password_correct = True
            st.rerun()  # Reload app to show content
        elif password_input:
            st.sidebar.error("Incorrect Password")
            
    return st.session_state.password_correct

if not check_password():
    st.stop()  # Stop execution if password is wrong
# ---------------------------

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
                st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
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