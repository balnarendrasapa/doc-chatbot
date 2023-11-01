import streamlit as st


def app():
    st.set_page_config(page_title="Chat with Documents")
    Option = st.select_box("Select an option", ("HuggingFace", "OpenAI"))


app()

