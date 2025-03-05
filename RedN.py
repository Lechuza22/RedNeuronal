import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cargar los modelos entrenados
model_unidades = tf.keras.models.load_model("modelo_unidades.h5")
model_importe_canal = tf.keras.models.load_model("modelo_importe_canal.h5")
model_importe_provincia = tf.keras.models.load_model("modelo_importe_provincia.h5")

# Cargar los datos para obtener opciones
file_ventas = "ventas_Hoja1.csv"
df_ventas = pd.read_csv(file_ventas).dropna()

# Codificar variables categóricas
label_encoders = {}
categorical_cols = ["categoria", "canal", "provincia"]

for col in categorical_cols:
    le = LabelEncoder()
    df_ventas[col] = le.fit_transform(df_ventas[col])
    label_encoders[col] = le

# Normalizar las variables de salida
scaler = StandardScaler()
df_ventas[["unidades", "importe"]] = scaler.fit_transform(df_ventas[["unidades", "importe"]])

# Obtener opciones únicas para los selectores
categorias = list(label_encoders["categoria"].classes_)
canales = list(label_encoders["canal"].classes_)
provincias = list(label_encoders["provincia"].classes_)

# Configurar la app Streamlit
st.title("Redes Neuronales y Ventas")

menu = st.sidebar.selectbox("Selecciona una opción", ["Predicción por Categoría", "Predicción por Canal", "Predicción por Provincia"])

if menu == "Predicción por Categoría":
    st.subheader("Predicción de Unidades Vendidas por Categoría")
    categoria = st.selectbox("Selecciona una categoría", categorias)
    
    if st.button("Predecir"):
        cat_encoded = label_encoders["categoria"].transform([categoria])[0]
        prediction = model_unidades.predict(np.array([[cat_encoded]]))
        pred_units = scaler.inverse_transform([[prediction[0][0], 0]])[0][0]
        st.success(f"Unidades estimadas a vender: {pred_units:.2f}")

if menu == "Predicción por Canal":
    st.subheader("Predicción de Importe por Canal")
    canal = st.selectbox("Selecciona un canal", canales)
    
    if st.button("Predecir"):
        canal_encoded = label_encoders["canal"].transform([canal])[0]
        prediction = model_importe_canal.predict(np.array([[canal_encoded]]))
        pred_importe = scaler.inverse_transform([[0, prediction[0][0]]])[0][1]
        st.success(f"Importe estimado: ${pred_importe:.2f}")

if menu == "Predicción por Provincia":
    st.subheader("Predicción de Importe por Provincia")
    provincia = st.selectbox("Selecciona una provincia", provincias)
    
    if st.button("Predecir"):
        provincia_encoded = label_encoders["provincia"].transform([provincia])[0]
        prediction = model_importe_provincia.predict(np.array([[provincia_encoded]]))
        pred_importe = scaler.inverse_transform([[0, prediction[0][0]]])[0][1]
        st.success(f"Importe estimado: ${pred_importe:.2f}")

