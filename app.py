import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
warnings.filterwarnings("ignore")

from io import BytesIO
import streamlit as st
from streamlit import sidebar
from pypdf import PdfReader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

st.header("🗒️ NoteBot")

with sidebar:
    st.title("My Notes")
    file = st.file_uploader(
        label="Upload notes PDF and start asking questions",
        type="pdf"
    )

if file is not None:
    if isinstance(file, list):
        file = file[0]

    my_pdf = PdfReader(BytesIO(file.read()))
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)

    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"token": hf_token}
        )

    embedding = load_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding)

    user_query = st.text_input("Type your query here")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)
        context = "\n\n".join([doc.page_content for doc in matching_chunks])

        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            huggingfacehub_api_token=hf_token,
            max_new_tokens=300,
            temperature=0.7,
            task="conversational"
        )
        llm = ChatHuggingFace(llm=llm_endpoint)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "Answer the question based strictly on the provided context. "
                    "If the answer is not in the context, say you don't know.\n\n"
                    "Context:\n{context}"
                )
            ),
            ("human", "{question}"),
        ])

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Thinking..."):
            output = chain.invoke({"context": context, "question": user_query})
            st.write(output)