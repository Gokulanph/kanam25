import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import tempfile

# Set page title
st.set_page_config(page_title="Customer Spending Analyzer")
st.title("Customer Spending Analyzer")

with st.sidebar:
    st.title("Provide your API key")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

if not OPENAI_API_KEY:
    st.info("Enter your OpenAI API key to continue")
    st.stop()    

# Initialize session state for vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# File upload
uploaded_file = st.file_uploader("Upload Bank Statement (PDF)", type=['pdf'])

# Initialize OpenAI
llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

def process_pdf(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load and process the PDF
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    return vector_store

# Process uploaded file
if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing your bank statement..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    st.success("Bank statement processed successfully!")

# Only show the chat interface if the file has been processed
if st.session_state.vector_store is not None:
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        You are a financial assistant. 
        Answer any questions related to the provided 
        bank statement.
        {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Set up retrieval chain
    retriever = st.session_state.vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)
    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Set up chat history
    history_for_chain = StreamlitChatMessageHistory()
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # Chat interface
    question = st.text_input("Ask a question about your bank statement:")
    if question:
        with st.spinner("Analyzing..."):
            response = chain_with_history.invoke(
                {"input": question},
                {"configurable": {"session_id": "abc123"}}
            )
            st.write(response['answer'])

# Add reset button
if st.session_state.vector_store is not None:
    if st.button("Upload New Statement"):
        st.session_state.vector_store = None
        st.experimental_rerun()