from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from pathlib import Path
import streamlit as st
from uuid import uuid4
import faiss
import time
import os

st.set_page_config(page_title="RAG Chatbot", page_icon=":robot_face:")

# Set Azure OpenAI environment variables
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]

# Initialize session state for navigation
st.session_state.setdefault("page", "upload")
st.session_state.setdefault("vectordb_created", False)
st.session_state.setdefault("docs", None)
st.session_state.setdefault("embedding", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("qa_chain_created", False)
st.session_state.setdefault("stored_files", [])
st.session_state.setdefault("selected_file", None)
st.session_state.setdefault("persist_directory", None)
st.session_state.setdefault("uploaded_files", [])
st.session_state.setdefault("option", "Few-Shot")

def load_pdf(uploaded_file: bytes) -> list:
    output_dir = Path("uploaded_files")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def create_docs(pages: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(pages)
    st.session_state.docs = docs
    return docs

def create_embeddings_vectore_db(docs: list, file_name: str) -> tuple:
    embedding = AzureOpenAIEmbeddings(model="text-embedding-3-large", azure_endpoint=st.secrets["AZURE_OPENAI_EMBEDDING_ENDPOINT"])
    st.session_state.embedding = embedding
    persist_directory = f"Vector_Database/{file_name}/faiss_index/"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        index = faiss.IndexFlatL2(len(embedding.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
        vector_store.save_local(persist_directory)
        st.session_state.vectordb_created = True
    else:
        vector_store = FAISS.load_local(persist_directory, embedding, allow_dangerous_deserialization=True)
        st.session_state.vectordb_created = True
    return vector_store, persist_directory

def create_conversation_chain(persist_directory: str=st.session_state.persist_directory) -> None:
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", api_version="2023-03-15-preview")
    def format_docs(docs: list) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    vectordb = FAISS.load_local(persist_directory, st.session_state.embedding, allow_dangerous_deserialization=True)
    template = get_prompt_template(st.session_state.option)
    custom_rag_prompt = PromptTemplate.from_template(template)
    qa_chain = (
        {"context": vectordb.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    st.session_state.qa_chain = qa_chain
    st.session_state.qa_chain_created = True
    
def ChangeFile(selected_file: str) -> None:
    st.session_state.messages =[]
    st.session_state.selected_file = selected_file
    file_path = Path("uploaded_files") / selected_file
    if file_path.exists():
        persist_directory = f"Vector_Database/{selected_file}/faiss_index/"
        if not os.path.exists(persist_directory):
            pages = load_pdf(file_path)
            docs = create_docs(pages)
            vector_store, persist_directory = create_embeddings_vectore_db(docs, selected_file)
        else:
            vector_store = FAISS.load_local(persist_directory, st.session_state.embedding, allow_dangerous_deserialization=True)
        st.session_state.persist_directory = persist_directory
        st.session_state.vectordb = vector_store
        create_conversation_chain(persist_directory)
        st.session_state.qa_chain_created = True

def get_prompt_template(option: str) -> str:
    if option == "Zero-Shot":
        return "{context} Question: {question} Helpful Answer:"
    elif option == "Few-Shot":
        return """Use the following pieces of context to answer the question at the end.
                  {context} Question: {question} Helpful Answer:"""
    elif option == "Custom":
        return st.session_state.input_text + "\n{context} Question: {question} Helpful Answer:"

@st.dialog("Configure a New Chatbot")
def new_chatbot() -> None:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file and uploaded_file.name not in st.session_state.stored_files:
        st.session_state.uploaded_files.append(uploaded_file)
        with st.spinner('Vector Database is being created'):
            pages = load_pdf(uploaded_file)
            docs = create_docs(pages)
            vector_store, persist_directory = create_embeddings_vectore_db(docs, uploaded_file.name)
            st.session_state.stored_files.append(uploaded_file.name)
        with st.spinner('Question Answer Chain is being created'):            
            st.session_state.persist_directory = persist_directory
            create_conversation_chain(st.session_state.persist_directory)
        st.session_state.selected_file = uploaded_file.name
        st.session_state.page = "chat"
        st.session_state.messages =[]
        st.success("File Uploaded Successfully")
        with st.spinner("Chatbot is ready"):
            time.sleep(2)
        st.rerun()
    elif uploaded_file and uploaded_file.name in st.session_state.stored_files:
        with st.spinner("File is already Available"):
            ChangeFile(uploaded_file.name)
            time.sleep(2)
        st.rerun()

# Sidebar for file upload and prompt selection
with st.sidebar:
    if st.button("New Chat", use_container_width=True):
        new_chatbot()

    if st.session_state.stored_files:
        selected_file = st.selectbox("Select a previously uploaded file", st.session_state.stored_files, index=st.session_state.stored_files.index(st.session_state.selected_file))
    else:
        selected_file = st.selectbox("Select a previously uploaded file", st.session_state.stored_files)
        
    if selected_file and selected_file != st.session_state.selected_file:
        ChangeFile(selected_file)

    if st.session_state.vectordb_created:
        with st.expander("Choose Prompt Type", expanded=True):
            st.session_state.option = st.radio("Select an option:", ["Few-Shot", "Zero-Shot", "Custom"], index=0, on_change=create_conversation_chain)
            if st.session_state.option == "Custom":
                st.session_state.input_text = st.text_area("Enter custom text:", height=150)
                create_conversation_chain()

# Main content (chat interface)
if st.session_state.page == "chat":
    st.title("ChatBot")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your question?"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = st.session_state.qa_chain.invoke(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})