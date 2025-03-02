import os
import tempfile
import time
from typing_extensions import List, TypedDict
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Research paper Chatbot", layout="wide", page_icon="👨‍🎓")
st.title("💬 Research paper Chatbot")

# API Key Setup
global GROQ_API_KEY, LANGSMITH_API_KEY


def get_api_key(env_var: str, secret_key: str, sidebar_label: str, placeholder: str) -> str:
    """Retrieve API key from environment variables, Streamlit secrets, or user input."""
    api_key = os.getenv(env_var) or st.secrets.get(secret_key)  # Prioritize environment and secrets
    if api_key:
        st.sidebar.success(f"{sidebar_label} loaded successfully ✅", icon="🔑")
    else:
        api_key = st.sidebar.text_input(
            label=f"#### Enter {sidebar_label} 👇",
            placeholder=placeholder,
            type="password",
            key=env_var
        )
        if api_key:
            st.sidebar.success(f"{sidebar_label} saved! ✅", icon="🔑")
    return api_key


with st.sidebar:
    st.title("💬 Research paper Chatbot")
    GROQ_API_KEY = get_api_key(
        "GROQ_API_KEY",
        "GROQ_API_KEY",
        "Groq API Key",
        "Paste your GROQ API key, gsk-"
    )
    LANGSMITH_API_KEY = get_api_key(
        "LANGSMITH_API_KEY",
        "LANGSMITH_API_KEY",
        "LangSmith API Key",
        "Paste your LangSmith API key, ls-"
    )

    if GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    if LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGSMITH_TRACING"] = "true"


# Sidebar settings
st.sidebar.header("⚙️ Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

# Load embeddings and LLM
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
if "llm" not in st.session_state:
    props = {
        "temperature": temperature,
        "timeout": 3,
    }
    st.session_state.llm = init_chat_model(
        model="llama3-8b-8192",
        model_provider="groq",
        **props
    )
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = embedding_function


@st.cache_resource
def chunk_and_index(path: str):
    loader = PyMuPDFLoader(path, extract_tables='markdown')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
    chunks = text_splitter.split_documents(docs)
    return FAISS.from_documents(chunks, st.session_state['embeddings'])


# Define prompt template
template = ChatPromptTemplate([
    ("system", """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, and ask to reframe the question.
    Keep the answer concise and always give responses in bullet points.
    At the end give a short summary of your response.
    Make sure to give your response in markdown format.
    Context: {context}
    """),
    ("human", "{question}"),
])


# Define chatbot state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define retrieve function
def retrieve(state: State):
    if st.session_state.vector_store:
        retrieved_docs = st.session_state.vector_store.similarity_search(state["question"], k=3)
    else:
        retrieved_docs = []
    return {"context": retrieved_docs}


# Define generate function
def generate(state: State):
    docs_content = "\n\n".join(
        doc.page_content
        for doc in state["context"]
    )
    msgs = template.invoke(
        {
            "question": state["question"],
            "context": docs_content
        }
    )
    ans = st.session_state.llm.invoke(msgs)
    return {"answer": ans.content}


# Build LangGraph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Sidebar settings
st.sidebar.header("📂 Upload the Research paper")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    st.sidebar.success("File uploaded", icon='🙌')
    begin_chunk = st.sidebar.button("Index the file")
    # create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    if begin_chunk and st.session_state.vector_store is None:
        st.sidebar.success("Processing file, please wait...", icon='🔄')
        st.session_state.vector_store = chunk_and_index(temp_file_path)
        st.sidebar.success("File processed...", icon='🙌')

# User input and chatbot response
user_input = st.chat_input("Ask me anything...")
if user_input:
    with st.chat_message("🧑"):
        st.write(user_input)
    response = graph.invoke({"question": user_input})
    ai_response = response["answer"]
    with st.chat_message("🤖"):
        st.write(ai_response)
        # response_container = st.empty()
        # full_response = ""
        # for chunk in ai_response.split():
        #     full_response += chunk + " "
        #     response_container.markdown(full_response + "▌")
        #     time.sleep(0.05)
