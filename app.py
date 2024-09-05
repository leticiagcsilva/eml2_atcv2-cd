import streamlit as st
import pandas as pd
from src.model import load_model, predict

model = load_model('models/model.pkl')

st.title('Previsão de Ações da AAPL')

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if st.button("Carregar arquivo de exemplo"):
    uploaded_file = 'data/sample.csv'

if uploaded_file is not None:
    if isinstance(uploaded_file, str):  
        data = pd.read_csv(uploaded_file)
    else:  
        data = pd.read_csv(uploaded_file)

    predictions = predict(model, data)
    st.write(predictions)
