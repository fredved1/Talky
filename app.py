import streamlit as st
import PyPDF2
import docx
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import openai
import os
from dotenv import load_dotenv
import time
from langsmith import Client
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Load environment variables
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if api_key is None:
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
    st.stop()

# Initialize ChatOpenAI with the API key and the model
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=2000, api_key=api_key)

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())

@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content

pipeline("Hello, world!")
# Out:  Hello there! How can I assist you today?



# Wrap the openai client for tracing
openai_client = wrap_openai(openai.Client())

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
        time.sleep(0.05)  # Faster for UX, but adjust as necessary
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
    
    @traceable  # Auto-trace this function
    def create_conversation_chain():
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(),
            memory=memory
        )

    conversation = create_conversation_chain()

    # Conversation interface
    st.subheader("Chat with your document")

    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        @traceable  # Auto-trace this function
        def get_answer(question):
            result = conversation({"question": question})
            return result['answer']

        answer = get_answer(user_question)
        st.session_state.chat_history.append(f"You: {user_question}")
        st.session_state.chat_history.append(f"Chatbot: {answer}")

    for chat in st.session_state.chat_history:
        st.write(chat)
