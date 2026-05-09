import streamlit as st
import pandas as pd
import pickle

# Configuração da página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

# 1. Carregar o modelo que você baixou do Colab
@st.cache_resource
def load_model():
    with open('modelo_final.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

st.title("🩺 Sistema de Apoio ao Diagnóstico - Obesidade")
st.markdown("Preencha os dados do paciente na barra lateral para realizar a predição.")

# 2. Criar a interface de entrada na barra lateral
with st.sidebar:
    st.header("Dados do Paciente")
    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 14, 61, 25)
    height = st.number_input("Altura (m)", 1.40, 2.20, 1.70)
    weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
    family_history = st.radio("Histórico Familiar de Sobrepeso?", ["yes", "no"])
    favc = st.radio("Consome alimentos calóricos frequentemente?", ["yes", "no"])
    fcvc = st.slider("Frequência de consumo de vegetais (1-3)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais", 1, 4, 3)
    caec = st.selectbox("Consumo de alimentos entre refeições", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.radio("Fumante?", ["yes", "no"])
    ch2o = st.slider("Consumo de água diário (1-3)", 1, 3, 2)
    scc = st.radio("Monitora o consumo de calorias?", ["yes", "no"])
    faf = st.slider("Atividade física semanal (0-3)", 0, 3, 1)
    tue = st.slider("Uso de dispositivos eletrônicos (0-2)", 0, 2, 1)
    calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Meio de transporte habitual", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# 3. Processar os dados e realizar a previsão
if st.button("Realizar Diagnóstico"):
    # Criar DataFrame com as entradas
    input_dict = {
        'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc,
        'NCP': ncp, 'CH2O': ch2o, 'FAF': faf, 'TUE': tue,
        'Gender_Male': 1 if gender == "Male" else 0,
        'family_history_yes': 1 if family_history == "yes" else 0,
        'FAVC_yes': 1 if favc == "yes" else 0,
        'CAEC_Frequently': 1 if caec == "Frequently" else 0,
        'CAEC_Sometimes': 1 if caec == "Sometimes" else 0,
        'CAEC_no': 1 if caec == "no" else 0,
        'SMOKE_yes': 1 if smoke == "yes" else 0,
        'SCC_yes': 1 if scc == "yes" else 0,
        'CALC_Frequently': 1 if calc == "Frequently" else 0,
        'CALC_Sometimes': 1 if calc == "Sometimes" else 0,
        'CALC_no': 1 if calc == "no" else 0,
        'MTRANS_Bike': 1 if mtrans == "Bike" else 0,
        'MTRANS_Motorbike': 1 if mtrans == "Motorbike" else 0,
        'MTRANS_Public_Transportation': 1 if mtrans == "Public_Transportation" else 0,
        'MTRANS_Walking': 1 if mtrans == "Walking" else 0
    }
    
    input_df = pd.DataFrame([input_dict])

    # Ajustar colunas para o modelo
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    # Prever
    res = model.predict(input_df)[0]
    
    # Mapeamento do resultado
    labels = ["Peso Insuficiente", "Peso Normal", "Obesidade Tipo I", "Obesidade Tipo II", "Obesidade Tipo III", "Sobrepeso Nível I", "Sobrepeso Nível II"]
    
    st.subheader("Resultado do Diagnóstico:")
    st.header(f"➡️ {labels[res]}")