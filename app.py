import streamlit as st


def app():
    st.set_page_config(page_title="Chat with AI", page_icon="🤖", layout="centered")
    st.title("Chat with AI")
    Option = st.selectbox(
        "Select an option",
        ("Select", "HuggingFace", "OpenAI")
    )
    if Option != "Select":
        doc = st.file_uploader("Upload a document", type=["pdf"])


if __name__ == "__main__":
    app()
