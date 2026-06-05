import os
import warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
warnings.filterwarnings("ignore")

from io import BytesIO
import streamlit as st
from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("HF_TOKEN not found in .env")
    st.stop()

st.header("🗒️ NoteBot")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader(
        label="Upload a notes PDF and start asking questions",
        type="pdf"
    )

# ── Cached helpers ────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"token": hf_token}
    )

@st.cache_resource
def load_llm():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",   # supported by most providers
        huggingfacehub_api_token=hf_token,
        max_new_tokens=300,
        temperature=0.5,
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm_endpoint)

# Cache the vector store per uploaded file (keyed by filename + size)
@st.cache_resource
def build_vector_store(file_key: str, raw_bytes: bytes):
    pdf = PdfReader(BytesIO(raw_bytes))
    text = "".join(page.extract_text() or "" for page in pdf.pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks, embeddings)

# ── Main logic ────────────────────────────────────────────
if file is not None:
    raw = file.read()
    file_key = f"{file.name}_{len(raw)}"          # stable cache key

    with st.spinner("Processing PDF..."):
        vector_store = build_vector_store(file_key, raw)

    user_query = st.text_input("Type your query here")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query, k=3)
        context = "\n\n".join(doc.page_content for doc in matching_chunks)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Answer the question based strictly on the provided context. "
                "If the answer is not in the context, say you don't know.\n\n"
                "Context:\n{context}"
            ),
            ("human", "{question}"),
        ])

        try:
            llm = load_llm()
            chain = prompt | llm | StrOutputParser()

            with st.spinner("Thinking..."):
                output = chain.invoke({"context": context, "question": user_query})
                # Some HF models echo the prompt — strip everything before [/INST] if present
                if "[/INST]" in output:
                    output = output.split("[/INST]")[-1].strip()
                st.write(output)

        except Exception as e:
            st.error(f"LLM error: {e}")