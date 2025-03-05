import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

# Cargar los datos
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

# Separar datos para los 3 modelos
X_categoria = df_ventas[["categoria"]]
y_unidades = df_ventas["unidades"]

X_canal = df_ventas[["canal"]]
y_importe_canal = df_ventas["importe"]

X_provincia = df_ventas[["provincia"]]
y_importe_provincia = df_ventas["importe"]

# Dividir en conjuntos de entrenamiento y prueba
X_cat_train, X_cat_test, y_uni_train, y_uni_test = train_test_split(X_categoria, y_unidades, test_size=0.2, random_state=42)
X_can_train, X_can_test, y_imp_can_train, y_imp_can_test = train_test_split(X_canal, y_importe_canal, test_size=0.2, random_state=42)
X_prov_train, X_prov_test, y_imp_prov_train, y_imp_prov_test = train_test_split(X_provincia, y_importe_provincia, test_size=0.2, random_state=42)

# Función para crear el modelo
def build_model():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(1,)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")  # Salida continua para regresión
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Entrenar y guardar modelos si no existen
if not os.path.exists("modelo_unidades.h5"):
    model_unidades = build_model()
    model_unidades.fit(X_cat_train, y_uni_train, epochs=10, batch_size=32, validation_data=(X_cat_test, y_uni_test), verbose=1)
    model_unidades.save("modelo_unidades.h5")

if not os.path.exists("modelo_importe_canal.h5"):
    model_importe_canal = build_model()
    model_importe_canal.fit(X_can_train, y_imp_can_train, epochs=10, batch_size=32, validation_data=(X_can_test, y_imp_can_test), verbose=1)
    model_importe_canal.save("modelo_importe_canal.h5")

if not os.path.exists("modelo_importe_provincia.h5"):
    model_importe_provincia = build_model()
    model_importe_provincia.fit(X_prov_train, y_imp_prov_train, epochs=10, batch_size=32, validation_data=(X_prov_test, y_imp_prov_test), verbose=1)
    model_importe_provincia.save("modelo_importe_provincia.h5")

# Cargar los modelos entrenados
def cargar_modelo(ruta):
    if os.path.exists(ruta):
        return tf.keras.models.load_model(ruta)
    else:
        st.error(f"⚠️ No se encontró el modelo {ruta}. Asegúrate de haberlo entrenado correctamente.")
        return None

model_unidades = cargar_modelo("modelo_unidades.h5")
model_importe_canal = cargar_modelo("modelo_importe_canal.h5")
model_importe_provincia = cargar_modelo("modelo_importe_provincia.h5")

# Funciones para predicciones
def predecir_unidades(categoria):
    cat_encoded = label_encoders["categoria"].transform([categoria])[0]
    prediction = model_unidades.predict(np.array([[cat_encoded]]))
    return scaler.inverse_transform([[prediction[0][0], 0]])[0][0]

def predecir_importe_canal(canal):
    canal_encoded = label_encoders["canal"].transform([canal])[0]
    prediction = model_importe_canal.predict(np.array([[canal_encoded]]))
    return scaler.inverse_transform([[0, prediction[0][0]]])[0][1]

def predecir_importe_provincia(provincia):
    provincia_encoded = label_encoders["provincia"].transform([provincia])[0]
    prediction = model_importe_provincia.predict(np.array([[provincia_encoded]]))
    return scaler.inverse_transform([[0, prediction[0][0]]])[0][1]

# Configurar la app Streamlit
st.title("Redes Neuronales y Ventas")

menu = st.sidebar.selectbox("Selecciona una opción", ["Predicción por Categoría", "Predicción por Canal", "Predicción por Provincia"])

if menu == "Predicción por Categoría":
    st.subheader("Predicción de Unidades Vendidas por Categoría")
    categoria = st.selectbox("Selecciona una categoría", list(label_encoders["categoria"].classes_))
    if st.button("Predecir"):
        pred_units = predecir_unidades(categoria)
        st.success(f"Unidades estimadas a vender: {pred_units:.2f}")

if menu == "Predicción por Canal":
    st.subheader("Predicción de Importe por Canal")
    canal = st.selectbox("Selecciona un canal", list(label_encoders["canal"].classes_))
    if st.button("Predecir"):
        pred_importe = predecir_importe_canal(canal)
        st.success(f"Importe estimado: ${pred_importe:.2f}")

if menu == "Predicción por Provincia":
    st.subheader("Predicción de Importe por Provincia")
    provincia = st.selectbox("Selecciona una provincia", list(label_encoders["provincia"].classes_))
    if st.button("Predecir"):
        pred_importe = predecir_importe_provincia(provincia)
        st.success(f"Importe estimado: ${pred_importe:.2f}")
