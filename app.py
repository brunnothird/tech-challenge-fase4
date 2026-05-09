import streamlit as st
import pandas as pd
import pickle

# Configuração da página
st.set_page_config(page_title="Predição de Obesidade", layout="wide")

# 1. Carregar o modelo
@st.cache_resource
def load_model():
    with open('modelo_final.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

st.title("🩺 Sistema de Apoio ao Diagnóstico - Obesidade")
st.markdown("Preencha os dados do paciente na barra lateral para realizar a predição baseada em IA.")

# 2. Criar a interface de entrada na barra lateral (TRADUZIDA E CORRIGIDA)
with st.sidebar:
    st.header("Dados do Paciente")
    
    # Tradução de Gênero
    gender_pt = st.selectbox("Gênero", ["Masculino", "Feminino"])
    gender = "Male" if gender_pt == "Masculino" else "Female"
    
    age = st.slider("Idade", 14, 61, 25)
    height = st.number_input("Altura (m)", 1.40, 2.20, 1.70)
    weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
    
    # Tradução de Sim/Não
    family_history_pt = st.radio("Histórico Familiar de Sobrepeso?", ["Sim", "Não"])
    family_history = "yes" if family_history_pt == "Sim" else "no"
    
    favc_pt = st.radio("Consome alimentos calóricos frequentemente?", ["Sim", "Não"])
    favc = "yes" if favc_pt == "Sim" else "no"
    
    # Ajuste: Vegetais por semana (Escala original era 1-3, ajustamos a proporção para 0-7)
    fcvc_week = st.slider("Frequência de consumo de vegetais (Dias por semana)", 0, 7, 3)
    fcvc = (fcvc_week / 7) * 2 + 1 # Converte escala 0-7 para 1-3 do modelo
    
    ncp = st.slider("Número de refeições principais ao dia", 1, 4, 3)

    
    smoke_pt = st.radio("Fumante?", ["Sim", "Não"])
    smoke = "yes" if smoke_pt == "Sim" else "no"
    
    # Ajuste: Água em Litros (Escala original era 1-3, ajustamos 0-5 litros para essa faixa)
    ch2o_litros = st.slider("Consumo de água diário (Litros)", 0.0, 5.0, 2.0)
    ch2o = (ch2o_litros / 5) * 2 + 1 # Converte escala 0-5 para 1-3 do modelo
    
    scc_pt = st.radio("Monitora o consumo de calorias?", ["Sim", "Não"])
    scc = "yes" if scc_pt == "Sim" else "no"
    
    # Ajuste: Atividade física semanal de 0 a 7
    faf_days = st.slider("Atividade física semanal (Dias por semana)", 0, 7, 2)
    faf = (faf_days / 7) * 3 # Converte escala 0-7 para 0-3 do modelo
    
    # Pergunta de eletrônicos removida conforme solicitado (TUE será enviado como 1 fixo para não quebrar o modelo)
    tue = 1.0 
    
    # Tradução Álcool
    calc_options = {"Não consome": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
    calc_pt = st.selectbox("Consumo de álcool", list(calc_options.keys()))
    calc = calc_options[calc_pt]
    
    # Tradução Transporte
    mtrans_options = {
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Automóvel": "Automobile",
        "Motocicleta": "Motorbike",
        "Bicicleta": "Bike"
    }
    mtrans_pt = st.selectbox("Meio de transporte habitual", list(mtrans_options.keys()))
    mtrans = mtrans_options[mtrans_pt]

# 3. Processar e Prever
if st.button("Realizar Diagnóstico"):
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

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    res = model.predict(input_df)[0]
    
    labels = ["Peso Insuficiente", "Peso Normal", "Obesidade Tipo I", "Obesidade Tipo II", "Obesidade Tipo III", "Sobrepeso Nível I", "Sobrepeso Nível II"]
    
    st.subheader("Resultado do Diagnóstico:")
    st.success(f"O paciente foi classificado com: **{labels[res]}**")
    