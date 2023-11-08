import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
import time
# from langchain import HuggingFaceHub

if "responses" not in st.session_state:
    st.session_state.responses = []

if "questions" not in st.session_state:
    st.session_state.questions = []


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
                llm = OpenAI(open_api_key=API) if Option == "OpenAI" else (
                    HuggingFaceHub(
                        repo_id="tiiuae/falcon-7b-instruct",
                        model_kwargs={
                            "temperature": 0.5,
                            "max_new_tokens": 500
                        },
                        huggingfacehub_api_token=API,
                    )
                )
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=faiss_index.as_retriever(
                        search_type="mmr",
                        search_kwargs={'fetch_k': 10}),
                    return_source_documents=True
                )

                container = st.container()
                st.write("Ask Your Question Here")
                question = st.text_input(
                    "Ask your question here",
                    label_visibility="collapsed"
                )
                with container:
                    with st.chat_message("assistant"):
                        st.write("How can I help you?")

                    if question != "":
                        response = qa(question)
                        st.session_state.responses.append(response)
                        st.session_state.questions.append(question)

                        for i in range(len(st.session_state.responses)):
                            with st.chat_message("user"):
                                st.write(st.session_state.questions[i - 1])

                            with st.chat_message("assistant"):
                                result = st.session_state.responses[i]
                                st.write(result['result'])
                                st.write("Source documents: "
                                         "(Most relevant are first)")
                                for i in result['source_documents']:
                                    with st.expander(
                                        "Page " + str(i.metadata['page'])
                                    ):
                                        st.write(i.page_content)
                                st.divider()


if __name__ == "__main__":
    app()
