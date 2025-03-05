import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

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
models["unidades_por_categoria"].fit(X_cat_train, y_uni_train, epochs=10, batch_size=32, validation_data=(X_cat_test, y_uni_test), verbose=1)
models["importe_por_canal"].fit(X_can_train, y_imp_can_train, epochs=10, batch_size=32, validation_data=(X_can_test, y_imp_can_test), verbose=1)
models["importe_por_provincia"].fit(X_prov_train, y_imp_prov_train, epochs=10, batch_size=32, validation_data=(X_prov_test, y_imp_prov_test), verbose=1)

# Evaluar modelos
results = {
    "unidades_por_categoria": models["unidades_por_categoria"].evaluate(X_cat_test, y_uni_test, verbose=0),
    "importe_por_canal": models["importe_por_canal"].evaluate(X_can_test, y_imp_can_test, verbose=0),
    "importe_por_provincia": models["importe_por_provincia"].evaluate(X_prov_test, y_imp_prov_test, verbose=0)
}

print("Resultados de evaluación:")
print(results)


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

