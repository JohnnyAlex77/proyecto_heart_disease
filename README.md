# Heart Disease Prediction Model
![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
## 📋 Descripción del Proyecto
Este proyecto desarrolla un **modelo de Machine Learning** para predecir la presencia de enfermedad cardíaca en pacientes utilizando datos clínicos de rutina. El modelo está basado en el algoritmo **Random Forest** y alcanza una precisión del **90.16%**.
El proyecto fue desarrollado como parte de la asignatura **Tendencias de Inteligencia Artificial Aplicada** y está diseñado para ser una herramienta de apoyo al diagnóstico clínico en atención primaria.
## 📊 Resultados Principales
| Métrica | Valor | Interpretación Clínica |
|---------|-------|------------------------|
| Accuracy | 90.16% | Clasifica correctamente 9 de cada 10 pacientes |
| Recall (Sensibilidad) | 88.57% | Detecta al 88.6% de los pacientes realmente enfermos |
| Especificidad | 92.31% | Identifica correctamente al 92.3% de los pacientes sanos |
| F1-Score | 89.86% | Media armónica entre precisión y recall |
| AUC-ROC | 0.94 | Excelente capacidad de separar clases |
### Matriz de Confusión
| | Predijo: Sano | Predijo: Enfermo |
|------------|---------------|------------------|
| Real: Sano | 24 (VP) | 2 (FP) |
| Real: Enfermo | 4 (FN) | 31 (VN) |
## 🔬 Top 5 Variables Más Importantes
| Variable | Descripción | Importancia |
|----------|-------------|-------------|
| cp | Tipo de dolor de pecho | 18.2% |
| thalach | Frecuencia cardíaca máxima | 15.7% |
| ca | Número de vasos principales coloreados | 14.3% |
| oldpeak | Depresión del segmento ST | 12.8% |
| exang | Angina inducida por ejercicio | 9.5% |
## 📁 Estructura del Proyecto

heart_disease/  
│  
├── heart_disease_model.py # Código principal del modelo  
├── requirements.txt # Dependencias del proyecto  
├── README.md # Este archivo  
├── .gitignore # Archivos ignorados por Git  
│  
├── heart.csv # Dataset (1000 registros, 13 variables)  
│  
├── 1_matriz_correlacion.png # Matriz de correlación entre variables  
├── 2_matriz_confusion.png # Matriz de confusión del modelo  
├── 3_curva_roc.png # Curva ROC (AUC = 0.94)  
├── 4_importancia_variables.png # Importancia de variables clínicas  
│  
├── heart_disease_model.pkl # Modelo entrenado (Random Forest)  
└── scaler.pkl # Escalador para normalizar datos

## 🚀 Instalación y Ejecución
### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/JohnnyAlex77/heart_disease.git
cd heart_disease
```
### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
```
### Paso 3: Ejecutar el modelo

```bash
python heart_disease_model.py
```
## 📈 Visualizaciones Generadas

El script genera automáticamente 4 visualizaciones:

|Archivo|Descripción|
|---|---|
|1_matriz_correlacion.png|Correlaciones entre todas las variables clínicas|
|2_matriz_confusion.png|Matriz de confusión del modelo Random Forest|
|3_curva_roc.png|Curva ROC con valor AUC|
|4_importancia_variables.png|Ranking de variables predictoras|

## 🧪 Prueba con un Paciente Nuevo

El código incluye un ejemplo de predicción para un nuevo paciente:

```python

# Paciente de ejemplo: 55 años, hombre, colesterol alto, presión elevada
paciente_ejemplo = [[55, 1, 2, 145, 280, 0, 1, 150, 0, 1.5, 1, 0, 2]]
# El modelo devuelve: ALTO RIESGO (probabilidad: ~85%)
```
## 📊 Descripción del Dataset

El dataset utilizado es una versión ampliada del "Heart Disease Dataset" del repositorio UCI, con **1000 registros** y **14 columnas**:

|Variable|Descripción|Rango|
|---|---|---|
|age|Edad en años|29-77|
|sex|Género (1=masculino, 0=femenino)|0, 1|
|cp|Tipo de dolor de pecho|0-3|
|trestbps|Presión arterial en reposo (mm Hg)|94-200|
|chol|Colesterol sérico (mg/dl)|126-564|
|fbs|Glucosa en ayunas > 120 mg/dl|0, 1|
|restecg|Resultados electrocardiográficos|0-2|
|thalach|Frecuencia cardíaca máxima|71-202|
|exang|Angina inducida por ejercicio|0, 1|
|oldpeak|Depresión del segmento ST|0-6.2|
|slope|Pendiente del segmento ST|0-2|
|ca|Número de vasos principales|0-3|
|thal|Tipo de talasemia|3, 6, 7|
|target|Presencia de enfermedad (1=Sí, 0=No)|0, 1|
## 🤖 Modelos Comparados

Se entrenaron y compararon 4 algoritmos de clasificación:

|Modelo|Accuracy|AUC-ROC|
|---|---|---|
|Regresión Logística|85.25%|0.88|
|Árbol de Decisión|81.97%|0.76|
|Random Forest|90.16%|0.94|
|SVM|81.97%|0.85|

**Random Forest fue seleccionado por:** mayor precisión, robustez ante datos mixtos y capacidad de identificar variables importantes.

## ⚠️ Limitaciones del Modelo

|Limitación|Descripción|
|---|---|
|Tamaño muestral reducido|Solo 1000 pacientes → puede no generalizar bien|
|Sesgo de género|~68% hombres → subrepresenta enfermedad en mujeres|
|Origen geográfico limitado|Datos principalmente de Cleveland, EE.UU.|
|Falta de variables clave|No incluye tabaquismo, diabetes, IMC, antecedentes familiares|
## 💡 Recomendaciones para Uso Clínico

|Acción|Prioridad|
|---|---|
|Validación externa con datos locales|Alta|
|Ajuste de umbral a 0.3 para aumentar sensibilidad|Alta|
|Estudio prospectivo en centro de salud|Media|
|Incorporar más variables (tabaquismo, diabetes)|Alta|
## 🔮 Perspectivas Futuras

|Plazo|Acción|
|---|---|
|Corto plazo|Herramienta de apoyo en atención primaria|
|Mediano plazo|Aplicación web con Streamlit|
|Largo plazo|Integración en historias clínicas electrónicas|
## 📚 Bibliografía

|#|Fuente|
|---|---|
|1|Organización Mundial de la Salud (OMS). (2021). _Enfermedades cardiovasculares (ECV)_.|
|2|UCI Machine Learning Repository. (1989). _Heart Disease Data Set_.|
|3|Breiman, L. (2001). Random Forests. _Machine Learning_, 45(1), 5–32.|
|4|Raschka, S. (2018). _Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning_.|
|5|Janowicz, A. (2020). _A guide to Scikit-learn's StandardScaler_. Towards Data Science.|
## 👥 Autores

|Nombre|Rol|
|---|---|
|Johnny Valladares|Código, configuración de Colab, entrenamiento de modelos|
|Jaime Jiménez|Análisis clínico, interpretación de resultados|
|Lorenzo Vargas|Documentación, video, presentación|

_Asignatura: Tendencias de Inteligencia Artificial Aplicada_  
*Sección: ETDI02-C12*  
_Docente: Claudio Ariel Valdebenito López_  
*Fecha: 06-06-2026*

## 📄 Licencia

Este proyecto es de carácter académico. El código está disponible para fines educativos y de investigación.

## 🔗 Enlaces

|Recurso|Enlace|
|---|---|
|Repositorio GitHub|[https://github.com/JohnnyAlex77/heart_disease](https://github.com/JohnnyAlex77/heart_disease)|
|Google Colab|[Insertar enlace del notebook interactivo]|