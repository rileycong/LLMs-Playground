import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Youtube Assistant 🤖")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label= "Enter the YouTube video URL",
            max_chars = 50
        )
        query = st.sidebar.text_area(
            label="Ask something about the video",
            max_chars=200,
            key="query"
        )

        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = lch.create_vectordb(youtube_url)
    response = lch.get_response(db=db, query=query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))
