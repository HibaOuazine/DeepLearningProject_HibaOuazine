import streamlit as st

st.title("Fruit Classification")
st.write("Upload an image of a fruit , and the model will predict its type !.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.sidebar.title("Hiba Ouazine")
st.sidebar.info(
    "Emsi2024 machine learning project"
)
st.sidebar.image("tofcv1.png",width=200)