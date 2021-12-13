import streamlit as st

def show():
    st.title("INTELIGENCIA ARTIFICIAL")
    st.write("\"La ciencia y la ingenieria de crear máquinas inteligentes, especialmente programas de computadoras inteligentes, que comprendan la inteligencia humano \".")
    st.write("-John McCarthy")
    from PIL import Image
    image = Image.open('IA.jpg')
    st.image(image)
    