# Semana 8 - Tito: clasificador de intenciones
# Marco Morquecho NC 1440

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("TITO - CLASIFICADOR DE INTENCIONES")

# Cargar el dataset
df = pd.read_csv("dataset_tito_v1.csv", encoding="utf-8-sig")

# Aceptar categoria con o sin acento
if "categoría" in df.columns:
    df = df.rename(columns={"categoría": "categoria"})

# Revisar que existan las columnas necesarias
if "texto" not in df.columns or "categoria" not in df.columns:
    print("El CSV debe tener las columnas texto y categoria")
    raise SystemExit

print("\nPreguntas del dataset:")
print(df.head())

print("\nCantidad por categoria:")
print(df["categoria"].value_counts())

# Separar preguntas y categorias
X = df["texto"].astype(str)
y = df["categoria"]

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Crear el modelo TF-IDF + Naive Bayes
modelo = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clasificador", MultinomialNB())
])

# Entrenar el modelo
modelo.fit(X_train, y_train)

print("\nModelo entrenado correctamente")

# Probar preguntas nuevas
preguntas = [
    "que necesito para una beca",
    "a que hora abre la biblioteca",
    "como me inscribo al siguiente semestre",
    "donde consulto mis calificaciones",
    "necesito orientacion escolar"
]

print("\nPRUEBAS DE TITO")
for pregunta in preguntas:
    categoria = modelo.predict([pregunta])[0]
    print(pregunta, "->", categoria)

# Guardar el modelo entrenado
joblib.dump(modelo, "modelo_tito_semana8.joblib")
print("\nModelo guardado como modelo_tito_semana8.joblib")

# Modo interactivo
while True:
    pregunta = input("\nPregunta para Tito (salir para terminar): ")

    if pregunta.lower() == "salir":
        print("Tito ha terminado")
        break

probabilidades = modelo.predict_proba([pregunta])[0]

mayor = max(probabilidades)

if mayor < 0.40:
    print("Tito: No entiendo tu pregunta. Puedes explicarla mejor.")
else:
    categoria = modelo.predict([pregunta])[0]
    print("Categoria:", categoria)