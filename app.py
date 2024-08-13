import streamlit as st
import PyPDF2
import docx
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import openai
import os
from dotenv import load_dotenv
import time

# Laad de omgevingsvariabelen uit het .env bestand
load_dotenv(dotenv_path="api.env")

# Haal de API-key op uit de omgevingsvariabelen
api_key = os.getenv("OPENAI_API_KEY")

# Controleer of de API-key is ingesteld
if api_key is None:
    st.error("OPENAI_API_KEY is niet ingesteld. Voeg deze toe aan je .env bestand.")
    st.stop()

# Initialiseer ChatOpenAI met de API-key en het nieuwe model
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=20000, api_key=api_key)

# Function to load PDF content
def load_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to load DOCX content
def load_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Streamlit interface
st.title("Document Chatbot")

# Upload document
uploaded_file = st.file_uploader("Upload your document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    # Show progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulate loading progress
    for percent_complete in range(100):
        time.sleep(0.1)
        progress_bar.progress(percent_complete + 1)
        status_text.text(f"Loading... {percent_complete + 1}%")

    # Load document
    if uploaded_file.type == "text/plain":
        raw_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        raw_text = load_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        raw_text = load_docx(uploaded_file)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_text(raw_text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create vector store
    db = FAISS.from_texts(texts, embeddings)

    # Create conversation chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_len=1000)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Use the initialized ChatOpenAI model
        retriever=db.as_retriever(),
        memory=memory
    )

    # Conversation interface
    st.subheader("Chat with your document")

    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        result = conversation({"question": user_question})
        st.session_state.chat_history.append(f"You: {user_question}")
        st.session_state.chat_history.append(f"Chatbot: {result['answer']}")

    for chat in st.session_state.chat_history:
        st.write(chat)