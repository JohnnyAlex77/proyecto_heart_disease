# Heart Disease Prediction Model
![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
## 📋 Descripción del Proyecto
Este proyecto desarrolla un **modelo de Machine Learning** para predecir la presencia de enfermedad cardíaca en pacientes utilizando datos clínicos de rutina. El modelo está basado en el algoritmo **Random Forest** y alcanza una sensibilidad del **96%**.
El proyecto fue desarrollado como parte de la asignatura **Tendencias de Inteligencia Artificial Aplicada** y está diseñado para ser una herramienta de apoyo al diagnóstico clínico en atención primaria.
## 📊 Resultados Principales
| Métrica | Valor | Interpretación Clínica |
|---------|-------|------------------------|
| **Accuracy** | 92.75% | Clasifica correctamente casi 93 de cada 100 pacientes |
| **Recall (Sensibilidad)** | 96.00% | Detecta al **96%** de los pacientes realmente enfermos |
| **Especificidad** | 84.21% | Identifica correctamente al 84.2% de los pacientes sanos |
| **F1-Score** | 95.05% | Media armónica entre precisión y recall |
| **AUC-ROC** | 0.985 | Capacidad discriminativa **casi perfecta** |
### Matriz de Confusión
| | Predijo: Sano | Predijo: Enfermo |
|------------|---------------|------------------|
| Real: Sano | 16 (VN) | 3 (FP) |
| Real: Enfermo | 2 (FN) | 48 (VP) |
## 🔬 Top 5 Variables Más Importantes
| Variable | Descripción | Importancia |
|----------|-------------|-------------|
| **age** | Edad en años | 24.99% |
| **thalach** | Frecuencia cardíaca máxima | 22.14% |
| **oldpeak** | Depresión del segmento ST | 14.17% |
| **trestbps** | Presión arterial en reposo | 12.17% |
| **slope** | Pendiente del segmento ST | 10.92% |
## 📁 Estructura del Proyecto

heart_disease/  
│  
├── heart_disease_model.py # Código principal del modelo  
├── requirements.txt # Dependencias del proyecto  
├── README.md # Este archivo  
├── .gitignore # Archivos ignorados por Git  
│  
├── heart.csv # Dataset (343 registros, 13 variables)  
│  
├── 1_matriz_correlacion.png # Matriz de correlación entre variables  
├── 2_matriz_confusion.png # Matriz de confusión del modelo  
├── 3_comparativa_modelos.png # Comparativa de Accuracy entre modelos  
├── 3_curva_roc.png # Curva ROC (AUC = 0.985)  
├── 4_importancia_variables.png # Importancia de variables clínicas  
│  
├── heart_disease_model.pkl # Modelo entrenado (Random Forest)  
└── scaler.pkl # Escalador para normalizar datos

text

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

El script genera automáticamente 5 visualizaciones:

|Archivo|Descripción|
|---|---|
|`1_matriz_correlacion.png`|Correlaciones entre todas las variables clínicas|
|`2_matriz_confusion.png`|Matriz de confusión del modelo Random Forest|
|`3_comparativa_modelos.png`|Comparativa de Accuracy entre los 4 modelos|
|`3_curva_roc.png`|Curva ROC con valor AUC (0.985)|
|`4_importancia_variables.png`|Ranking de variables predictoras|

## 🧪 Prueba con un Paciente Nuevo

El código incluye un ejemplo de predicción para un nuevo paciente:

```python
# Paciente de ejemplo: 55 años, hombre, colesterol alto, presión elevada
paciente_ejemplo = [[55, 1, 2, 145, 280, 0, 1, 150, 0, 1.5, 1, 0, 2]]
# El modelo devuelve: ALTO RIESGO (probabilidad: 75%)
```
## 📊 Descripción del Dataset

El dataset utilizado es una versión del "Heart Disease Dataset" del repositorio UCI, con **343 registros** y **14 columnas**:

|Variable|Descripción|Rango|
|---|---|---|
|age|Edad en años|34-71|
|sex|Género (1=masculino, 0=femenino)|0, 1|
|cp|Tipo de dolor de pecho|1, 2, 3|
|trestbps|Presión arterial en reposo (mm Hg)|100-172|
|chol|Colesterol sérico (mg/dl)|126-564|
|fbs|Glucosa en ayunas > 120 mg/dl|0, 1|
|restecg|Resultados electrocardiográficos|0, 1, 2|
|thalach|Frecuencia cardíaca máxima|71-202|
|exang|Angina inducida por ejercicio|0, 1|
|oldpeak|Depresión del segmento ST|0-6.2|
|slope|Pendiente del segmento ST|0, 1, 2|
|ca|Número de vasos principales|0-3|
|thal|Tipo de talasemia|1, 2, 3|
|**target**|**Presencia de enfermedad (1=Sí, 0=No)**|**0, 1**|

### Distribución de Clases

- **Enfermos (target=1):** 72.9%
- **Sanos (target=0):** 27.1%

## 🤖 Modelos Comparados

Se entrenaron y compararon 4 algoritmos de clasificación:

|Modelo|Accuracy|
|---|---|
|Regresión Logística|**94.20%**|
|Árbol de Decisión|91.30%|
|Random Forest|92.75%|
|SVM|**94.20%**|

**Random Forest fue seleccionado por:** capacidad de identificar variables importantes, robustez ante datos mixtos y control de sobreajuste.
## ⚠️ Limitaciones del Modelo

|Limitación|Descripción|
|---|---|
|Tamaño muestral reducido|Solo 343 pacientes → puede no generalizar bien|
|Sesgo de género|75.8% hombres → subrepresenta enfermedad en mujeres|
|Desbalance de clases|72.9% enfermos vs 27.1% sanos|
|Origen geográfico limitado|Datos sin especificar origen|
|Falta de variables clave|No incluye tabaquismo, diabetes, IMC, antecedentes familiares|
## 💡 Recomendaciones para Uso Clínico

|Acción|Prioridad|
|---|---|
|Validación externa con datos locales|Alta|
|Balanceo de clases (SMOTE)|Alta|
|Estudio prospectivo en centro de salud|Media|
|Incorporar más variables (tabaquismo, diabetes)|Alta|
|Reducir sesgo de género|Alta|

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
|Johnny Valladares|Código, configuración de Colab/VSCode, entrenamiento de modelos|
|Jaime Jiménez|Análisis clínico, interpretación de resultados|
|Lorenzo Vargas|Documentación, video, presentación, GitHub|

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
