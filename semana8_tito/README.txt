SEMANA 8 - TITO
Clasificacion de intenciones y evaluacion del modelo

Alumno: Marco Morquecho
NC: 1440
Proyecto: Tito - Asistente Escolar Inteligente por Voz
Materia: Procesamiento de Lenguaje Natural

ARCHIVOS:
- dataset_tito_v1.csv: dataset de ejemplo para entrenar y evaluar.
- dataset_tito_v2.csv: copia inicial para agregar mejoras.
- clase1_clasificador.py: entrena y prueba el clasificador.
- clase2_evaluacion.py: calcula accuracy, reporte, matriz y errores.
- modelo_tito_semana8.joblib: se crea al ejecutar clase1_clasificador.py.

CATEGORIAS DE EJEMPLO:
- becas
- biblioteca
- inscripciones
- orientacion
- control escolar
- otras

IMPORTANTE:
Los CSV incluidos contienen preguntas de ejemplo para que los programas
puedan ejecutarse. Si ya tienes el dataset de la Semana 7, conserva sus
preguntas y categorias y agrega mas ejemplos. Revisa que las columnas
se llamen texto y categoria (o categoría).

INSTALACION:
pip install pandas scikit-learn joblib

EJECUTAR:
python clase1_clasificador.py
python clase2_evaluacion.py

CONCLUSION:
Durante esta semana aprendi a utilizar TF-IDF junto con el clasificador
Multinomial Naive Bayes para identificar la categoria de diferentes
preguntas. Tambien aprendi a separar datos de entrenamiento y prueba,
calcular el accuracy, consultar el reporte de clasificacion y revisar
los errores mediante una matriz de confusion.

NOTA:
El accuracy y los errores deben copiarse de la ejecucion real del programa.
No se deben inventar resultados.
