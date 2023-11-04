import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
import time
# from langchain import HuggingFaceHub

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def app():
    PART_B = False
    part_a = st.empty()
    with part_a.container():
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
                    PART_B = True
                    Question = st.text_input("Ask a question")
                    if Question != "":
                        st.write(qa(Question))
        if PART_B:
            part_a.empty()
            if prompt := st.chat_input("What is up?"):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt
                    }
                )
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    assistant_response = qa(prompt)
                    for chunk in assistant_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response
                    }
                )


if __name__ == "__main__":
    app()
