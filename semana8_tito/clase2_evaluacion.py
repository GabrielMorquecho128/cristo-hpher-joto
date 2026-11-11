# Semana 8 - Tito: evaluacion del modelo
# Marco Morquecho NC 1440

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("TITO - EVALUACION DEL MODELO")

# Cargar dataset
df = pd.read_csv("dataset_tito_v1.csv", encoding="utf-8-sig")

if "categoría" in df.columns:
    df = df.rename(columns={"categoría": "categoria"})

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

# Crear y entrenar el modelo
modelo = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clasificador", MultinomialNB())
])

modelo.fit(X_train, y_train)

# Predecir las preguntas de prueba
y_pred = modelo.predict(X_test)

# Calcular accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nACCURACY DEL MODELO:")
print("Accuracy:", round(accuracy, 3))

# Reporte de clasificacion
print("\nREPORTE DE CLASIFICACION:")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusion
categorias = sorted(df["categoria"].unique())
matriz = confusion_matrix(y_test, y_pred, labels=categorias)

print("\nORDEN DE LAS CATEGORIAS:")
print(categorias)

print("\nMATRIZ DE CONFUSION:")
print(matriz)

# Mostrar errores reales
resultados = pd.DataFrame({
    "pregunta": X_test,
    "real": y_test,
    "predicha": y_pred
})

errores = resultados[resultados["real"] != resultados["predicha"]]

print("\nERRORES DEL MODELO:")
if len(errores) > 0:
    print(errores.to_string(index=False))
else:
    print("No hubo errores en este conjunto de prueba.")

print("\nEvaluacion terminada")
