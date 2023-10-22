import langchain_helper as lch
import streamlit as st

st.title("Movie Recommendation")

input_topic = st.sidebar.text_area("Movie topic", max_chars=50)
input_genre = st.sidebar.selectbox("Movie genre", ("romcom", "action", "musical", "drama", "documentation"))
enter = st.sidebar.button("See recommendations")

if enter:
    response = lch.recommend_movie(input_topic, input_genre)
    st.write(response['movie'])