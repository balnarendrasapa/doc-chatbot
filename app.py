import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


def app():
    st.set_page_config(
        page_title="Chat with AI",
        page_icon="🤖",
        layout="centered"
    )
    st.title("Chat with AI")
    Option = st.selectbox(
        "Select an option",
        ("Select", "HuggingFace", "OpenAI")
    )
    if Option != "Select":
        API = st.text_input("Enter your " + Option + " API key")
        if API != "":
            doc = st.file_uploader("Upload a document", type=["pdf"])
            if doc is not None:
                with open("doc.pdf", "wb") as f:
                    f.write(doc.getbuffer())
                loader = PyPDFLoader("doc.pdf")
                pages = loader.load_and_split()
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                faiss_index = FAISS.from_documents(pages, embeddings)
                llm = OpenAI(open_api_key=API)
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=faiss_index.as_retriever(
                        search_type="mmr",
                        search_kwargs={'fetch_k': 10}),
                    return_source_documents=True
                )
                Question = st.text_input("Ask a question")
                if Question != "":
                    st.write(qa.run(Question))


if __name__ == "__main__":
    app()
