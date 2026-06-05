"""
MODELO PREDICTIVO DE ENFERMEDADES CARDÍACAS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

print("=" * 60)
print("MODELO PREDICTIVO DE ENFERMEDADES CARDÍACAS")
print("=" * 60)

# ============================================
# 1. CARGAR DATOS
# ============================================
print("\n[1/9] Cargando datos...")
# IMPORTANTE: Cambia 'heart.csv' por el nombre real de tu archivo
df = pd.read_csv('heart.csv')  

print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nPrimeras 5 filas:")
print(df.head())

# ============================================
# 2. ANÁLISIS EXPLORATORIO (EDA)
# ============================================
print("\n[2/9] Análisis exploratorio...")

print("\n--- Estadísticas descriptivas ---")
print(df.describe())

print("\n--- Valores nulos ---")
print(df.isnull().sum())

print("\n--- Distribución de la variable objetivo ---")
distribucion = df['target'].value_counts(normalize=True) * 100
print(f"Pacientes sin enfermedad: {distribucion[0]:.1f}%")
print(f"Pacientes con enfermedad: {distribucion[1]:.1f}%")

# Guardar gráfico de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matriz de Correlación entre Variables Clínicas')
plt.tight_layout()
plt.savefig('1_matriz_correlacion.png', dpi=150)
plt.show()
print("Gráfico guardado: 1_matriz_correlacion.png")

# ============================================
# 3. PREPROCESAMIENTO
# ============================================
print("\n[3/9] Preprocesando datos...")

# Separar variables
X = df.drop('target', axis=1)
y = df['target']

# Dividir entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} registros")
print(f"Prueba: {X_test.shape[0]} registros")

# Escalar datos (necesario para Regresión Logística y SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Escalado completado")

# ============================================
# 4. ENTRENAR MODELOS
# ============================================
print("\n[4/9] Entrenando modelos...")

resultados = {}

# Regresión Logística
print("   • Regresión Logística...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
resultados['Regresión Logística'] = accuracy_score(y_test, y_pred_lr)

# Árbol de Decisión
print("   • Árbol de Decisión...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
resultados['Árbol de Decisión'] = accuracy_score(y_test, y_pred_dt)

# Random Forest
print("   • Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
resultados['Random Forest (Base)'] = accuracy_score(y_test, y_pred_rf)

# SVM
print("   • SVM...")
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
resultados['SVM'] = accuracy_score(y_test, y_pred_svm)

# Mostrar comparativa
print("\n" + "=" * 50)
print("COMPARATIVA DE MODELOS")
print("=" * 50)
for modelo, acc in resultados.items():
    print(f"{modelo:20} : {acc:.4f} ({acc*100:.2f}%)")

# ============================================
# FIGURA 3: GRÁFICO DE COMPARATIVA DE MODELOS
# ============================================
print("\n[4.5/9] Generando gráfico comparativo de modelos...")

# Datos para el gráfico (usando los resultados que ya calculaste)
modelos = list(resultados.keys())
accuracy_values = list(resultados.values())

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(modelos, accuracy_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('Accuracy (Precisión)')
plt.xlabel('Modelo')
plt.title('Comparativa de Accuracy entre Modelos')
plt.ylim(0.7, 1.0)  # Escala de 70% a 100%

# Agregar etiquetas con los valores sobre cada barra
for bar, acc in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.4f} ({acc*100:.2f}%)', ha='center', va='bottom', fontsize=10)

# Agregar línea de referencia al 90%
plt.axhline(y=0.90, color='gray', linestyle='--', alpha=0.7, label='Referencia 90%')
plt.legend()

plt.tight_layout()
plt.savefig('3_comparativa_modelos.png', dpi=150)
plt.show()
print("Gráfico guardado: 3_comparativa_modelos.png")

# ============================================
# 5. OPTIMIZAR RANDOM FOREST
# ============================================
print("\n[5/9] Optimizando Random Forest...")
best_rf = RandomForestClassifier(n_estimators=200, random_state=42)
best_rf.fit(X_train, y_train)
y_pred_best = best_rf.predict(X_test)
y_proba_best = best_rf.predict_proba(X_test)[:, 1]

accuracy_final = accuracy_score(y_test, y_pred_best)
print(f"Accuracy final: {accuracy_final*100:.2f}%")

# ============================================
# 6. MÉTRICAS DETALLADAS
# ============================================
print("\n[6/9] Generando métricas detalladas...")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_best, target_names=['Sano', 'Enfermo']))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sano', 'Enfermo'], 
            yticklabels=['Sano', 'Enfermo'])
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión - Random Forest')
plt.tight_layout()
plt.savefig('2_matriz_confusion.png', dpi=150)
plt.show()
print("Gráfico guardado: 2_matriz_confusion.png")

# Calcular métricas adicionales (para tu informe)
VP = cm[1, 1]  # Verdaderos Positivos (enfermos bien clasificados)
VN = cm[0, 0]  # Verdaderos Negativos (sanos bien clasificados)
FP = cm[0, 1]  # Falsos Positivos (sanos mal clasificados)
FN = cm[1, 0]  # Falsos Negativos (enfermos mal clasificados)

sensibilidad = VP / (VP + FN) * 100
especificidad = VN / (VN + FP) * 100
precision = VP / (VP + FP) * 100
f1_score = 2 * (precision * sensibilidad) / (precision + sensibilidad)

print(f"\n--- Métricas Clínicas ---")
print(f"Sensibilidad (Recall): {sensibilidad:.2f}%")
print(f"Especificidad: {especificidad:.2f}%")
print(f"Precisión: {precision:.2f}%")
print(f"F1-Score: {f1_score:.2f}%")

# ============================================
# 7. CURVA ROC
# ============================================
print("\n[7/9] Generando Curva ROC...")

fpr, tpr, _ = roc_curve(y_test, y_proba_best)
roc_auc = roc_auc_score(y_test, y_proba_best)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Clasificador Aleatorio')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curva ROC - Modelo Predictivo')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('3_curva_roc.png', dpi=150)
plt.show()
print(f"AUC-ROC: {roc_auc:.3f}")
print("Gráfico guardado: 3_curva_roc.png")

# ============================================
# 8. IMPORTANCIA DE VARIABLES
# ============================================
print("\n[8/9] Analizando importancia de variables...")

importances = best_rf.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({
    'Variable': features, 
    'Importancia': importances
}).sort_values('Importancia', ascending=False)

print("\n--- TOP 5 Variables más importantes ---")
print(feature_importance_df.head(5).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=feature_importance_df.head(10))
plt.title('Importancia de Variables Clínicas')
plt.xlabel('Importancia')
plt.ylabel('Variable')
plt.tight_layout()
plt.savefig('4_importancia_variables.png', dpi=150)
plt.show()
print("Gráfico guardado: 4_importancia_variables.png")

# ============================================
# 9. GUARDAR MODELO
# ============================================
print("\n[9/9] Guardando modelo...")

# Guardar modelo y escalador
joblib.dump(best_rf, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Modelo guardado: heart_disease_model.pkl")
print("Escalador guardado: scaler.pkl")

# ============================================
# 10. PRUEBA CON PACIENTE NUEVO
# ============================================
print("\n" + "=" * 60)
print("PRUEBA CON PACIENTE NUEVO")
print("=" * 60)

# Ejemplo: Paciente de 55 años, hombre, con varios factores de riesgo
# El orden de las columnas debe ser el mismo que en el entrenamiento
paciente_ejemplo = [[55, 1, 2, 145, 280, 0, 1, 150, 0, 1.5, 1, 0, 2]]

prediccion = best_rf.predict(paciente_ejemplo)
probabilidad = best_rf.predict_proba(paciente_ejemplo)[0][1]

print(f"Datos del paciente:")
print(f"  • Edad: 55 años")
print(f"  • Sexo: Hombre")
print(f"  • Colesterol: 280 mg/dl")
print(f"  • Presión arterial: 145 mm Hg")

if prediccion[0] == 1:
    print(f"\nRESULTADO: ALTO RIESGO")
    print(f"   Probabilidad de enfermedad: {probabilidad:.2%}")
    print(f"   Recomendación: Derivar a cardiología")
else:
    print(f"\nRESULTADO: BAJO RIESGO")
    print(f"   Probabilidad de enfermedad: {probabilidad:.2%}")
    print(f"   Recomendación: Seguimiento normal")

print("\n" + "=" * 60)
print("PROGRAMA COMPLETADO CON ÉXITO")
print("=" * 60)
print("\nArchivos generados:")
print("  • 1_matriz_correlacion.png")
print("  • 2_matriz_confusion.png")
print("  • 3_curva_roc.png")
print("  • 4_importancia_variables.png")
print("  • heart_disease_model.pkl")
print("  • scaler.pkl")