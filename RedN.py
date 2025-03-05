import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


# Cargar los archivos
file_ventas = "ventas_Hoja1.csv"
file_inventario = "mov_inventario_Hoja1.csv"

# Leer los datasets
df_ventas = pd.read_csv(file_ventas)
df_inventario = pd.read_csv(file_inventario)

# Mostrar información general de los datasets
df_ventas.info(), df_inventario.info()

# Eliminar registros con valores nulos en df_ventas
df_ventas_clean = df_ventas.dropna()

# Verificar que se eliminaron los valores nulos
df_ventas_clean.info()

# Cargar los archivos
file_ventas = "ventas_Hoja1.csv"
df_ventas = pd.read_csv(file_ventas)

# Eliminar registros con valores nulos
df_ventas_clean = df_ventas.dropna()

# Codificar variables categóricas
label_encoders = {}
categorical_cols = ["categoria", "canal", "provincia"]

for col in categorical_cols:
    le = LabelEncoder()
    df_ventas_clean[col] = le.fit_transform(df_ventas_clean[col])
    label_encoders[col] = le

# Normalizar las variables de salida
scaler = StandardScaler()
df_ventas_clean[["unidades", "importe"]] = scaler.fit_transform(df_ventas_clean[["unidades", "importe"]])

# Separar datos para los 3 modelos
X_categoria = df_ventas_clean[["categoria"]]
y_unidades = df_ventas_clean["unidades"]

X_canal = df_ventas_clean[["canal"]]
y_importe_canal = df_ventas_clean["importe"]

X_provincia = df_ventas_clean[["provincia"]]
y_importe_provincia = df_ventas_clean["importe"]

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

# Construir y entrenar modelos
models = {
    "unidades_por_categoria": build_model(),
    "importe_por_canal": build_model(),
    "importe_por_provincia": build_model()
}

# Entrenar modelos
models["unidades_por_categoria"].fit(X_cat_train, y_uni_train, epochs=50, batch_size=32, validation_data=(X_cat_test, y_uni_test), verbose=1)
models["importe_por_canal"].fit(X_can_train, y_imp_can_train, epochs=50, batch_size=32, validation_data=(X_can_test, y_imp_can_test), verbose=1)
models["importe_por_provincia"].fit(X_prov_train, y_imp_prov_train, epochs=50, batch_size=32, validation_data=(X_prov_test, y_imp_prov_test), verbose=1)

# Evaluar modelos
results = {
    "unidades_por_categoria": models["unidades_por_categoria"].evaluate(X_cat_test, y_uni_test, verbose=0),
    "importe_por_canal": models["importe_por_canal"].evaluate(X_can_test, y_imp_can_test, verbose=0),
    "importe_por_provincia": models["importe_por_provincia"].evaluate(X_prov_test, y_imp_prov_test, verbose=0)
}

print("Resultados de evaluación:")
print(results)


# Cargar los modelos
models = {
    "unidades_por_categoria": load_model("unidades_por_categoria"),
    "importe_por_canal": load_model("importe_por_canal"),
    "importe_por_provincia": load_model("importe_por_provincia")
}

# Cargar los datos para las opciones de selección
df_ventas = pd.read_csv("ventas_Hoja1.csv").dropna()

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

# Streamlit UI
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

# Desnormalizar las predicciones
pred_unidades_real = scaler.inverse_transform([[pred_unidades, 0]])[0][0]
pred_importe_canal_real = scaler.inverse_transform([[0, pred_importe_canal]])[0][1]
pred_importe_provincia_real = scaler.inverse_transform([[0, pred_importe_provincia]])[0][1]

# Mostrar resultados
st.write(f"**Unidades esperadas para la categoría '{categoria}':** {pred_unidades_real:.2f}")
st.write(f"**Importe esperado para el canal '{canal}':** {pred_importe_canal_real:.2f}")
st.write(f"**Importe esperado para la provincia '{provincia}':** {pred_importe_provincia_real:.2f}")
