import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Cargar el dataset
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

# Separar datos
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

# Función para crear modelos de redes neuronales
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
models_filenames = {
    "unidades_por_categoria": "model_unidades_por_categoria.h5",
    "importe_por_canal": "model_importe_por_canal.h5",
    "importe_por_provincia": "model_importe_por_provincia.h5"
}

models = {}

for key, filename in models_filenames.items():
    try:
        models[key] = load_model(filename)
    except:
        models[key] = build_model()
        if key == "unidades_por_categoria":
            models[key].fit(X_cat_train, y_uni_train, epochs=5, batch_size=16, validation_data=(X_cat_test, y_uni_test), verbose=0)
        elif key == "importe_por_canal":
            models[key].fit(X_can_train, y_imp_can_train, epochs=5, batch_size=16, validation_data=(X_can_test, y_imp_can_test), verbose=0)
        elif key == "importe_por_provincia":
            models[key].fit(X_prov_train, y_imp_prov_train, epochs=5, batch_size=16, validation_data=(X_prov_test, y_imp_prov_test), verbose=0)
        models[key].save(filename)

# Interfaz con Streamlit
st.title("Predicción de Ventas con Redes Neuronales")

# Menús desplegables
categoria = st.selectbox("Selecciona una categoría", label_encoders["categoria"].classes_)
provincia = st.selectbox("Selecciona una provincia", label_encoders["provincia"].classes_)
canal = st.selectbox("Selecciona un canal", label_encoders["canal"].classes_)

# Obtener valores codificados
categoria_encoded = label_encoders["categoria"].transform([categoria])[0]
provincia_encoded = label_encoders["provincia"].transform([provincia])[0]
canal_encoded = label_encoders["canal"].transform([canal])[0]

# Hacer predicciones
pred_unidades = models["unidades_por_categoria"].predict(np.array([[categoria_encoded]]))[0][0]
pred_importe_canal = models["importe_por_canal"].predict(np.array([[canal_encoded]]))[0][0]
pred_importe_provincia = models["importe_por_provincia"].predict(np.array([[provincia_encoded]]))[0][0]

# Desnormalizar predicciones
pred_unidades_real = scaler.inverse_transform([[pred_unidades, 0]])[0][0]
pred_importe_canal_real = scaler.inverse_transform([[0, pred_importe_canal]])[0][1]
pred_importe_provincia_real = scaler.inverse_transform([[0, pred_importe_provincia]])[0][1]

# Mostrar resultados
st.write(f"**Unidades esperadas para la categoría '{categoria}':** {pred_unidades_real:.2f}")
st.write(f"**Importe esperado para el canal '{canal}':** {pred_importe_canal_real:.2f}")
st.write(f"**Importe esperado para la provincia '{provincia}':** {pred_importe_provincia_real:.2f}")
