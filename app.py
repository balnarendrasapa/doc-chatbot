import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


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
        doc = st.file_uploader("Upload a document", type=["pdf"])
        if doc is not None:
            with open("doc.pdf", "wb") as f:
                f.write(doc.getbuffer())
            loader = PyPDFLoader("doc.pdf")
            pages = loader.load_and_split()
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            faiss_index = FAISS.from_documents(pages, embeddings)
            Question = st.text_input("Ask a question")
            if Question != "":
                docs = faiss_index.similarity_search(Question, k=5)
                for doc in docs:
                    st.write(
                        str(doc.metadata["page"]) + ":", doc.page_content[:300]
                    )


if __name__ == "__main__":
    app()
