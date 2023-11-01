import streamlit as st


def app():
    st.set_page_config(page_title="Chat with Documents")
    Option = st.selectbox("Select an option", ("Select", "HuggingFace", "OpenAI"))


app()

