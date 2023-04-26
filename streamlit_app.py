import streamlit as st

st.write(
    "Hello "
)

name = st.text_input("Name", key='name')

st.write(f"Hello, dear {name}")