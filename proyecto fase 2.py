import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score, 
                             f1_score, roc_auc_score)

# --- 1. CARGA DEL ARCHIVO ---
ruta_archivo = r"C:/Users/joelr/Documents/proyecto jesus/USvideos.csv"

try:
    df = pd.read_csv(ruta_archivo, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(ruta_archivo, encoding='latin-1')

print(f"Cargados {len(df)} registros.")

# --- 2. TRATAMIENTO ROBUSTO DE FECHAS (SOLUCIÓN AL ERROR) ---
# Convertimos a datetime directamente
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce', utc=True)
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce', utc=True)

# Eliminamos filas donde la fecha no se pudo procesar
df = df.dropna(subset=['publish_time', 'trending_date'])

# RESTA DIRECTA: Al ser objetos datetime, la resta genera un Timedelta de Pandas
# sobre el cual SÍ podemos usar .dt.days
df['days_to_trending'] = (df['trending_date'] - df['publish_time']).dt.days

# --- 3. LIMPIEZA Y FILTRADO ---
# Llenar nulos en descripción
df['description'] = df['description'].fillna("")

# Filtrar videos con fechas incoherentes (publicados después de ser tendencia)
# y filtrar outliers de vistas (según tu documento)
df = df[(df['days_to_trending'] >= 0) & (df['views'] < 1000000000)]

print(f"Registros tras limpieza: {len(df)}")

# --- 4. ANÁLISIS DE ESTACIONALIDAD (FIGURA 3 DEL DOC) ---
df['publish_day'] = df['publish_time'].dt.day_name()
dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(10, 5))
sns.countplot(x='publish_day', data=df, order=dias_orden, palette='viridis')
plt.title('Día de Publicación vs Frecuencia en Tendencias')
plt.show()

# --- 5. MODELADO DE MACHINE LEARNING ---
# Clasificación: Viralidad (> 1M vistas)
df['is_popular'] = (df['views'] > 1000000).astype(int)

# Variables (Features)
X = df[['likes', 'dislikes', 'comment_count', 'category_id']]
y_class = df['is_popular']
y_reg = df['days_to_trending']

# División 85% Train / 15% Test
X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
    X, y_class, y_reg, test_size=0.15, random_state=42
)

# Entrenamiento
print("\nEntrenando modelos...")
model_c = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train_c)
model_r = RandomForestRegressor(n_estimators=100).fit(X_train, y_train_r)

# --- 6. EVALUACIÓN DE MÉTRICAS ---
pred_c = model_c.predict(X_test)
pred_r = model_r.predict(X_test)

print(f"\n--- RESULTADOS CLASIFICACIÓN ---")
print(f"Accuracy: {accuracy_score(y_test_c, pred_c):.4f}")
print(f"F1-Score: {f1_score(y_test_c, pred_c):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test_c, model_c.predict_proba(X_test)[:, 1]):.4f}")

print(f"\n--- RESULTADOS REGRESIÓN ---")
print(f"MAE: {mean_absolute_error(y_test_r, pred_r):.2f} días")
print(f"R2 Score: {r2_score(y_test_r, pred_r):.4f}")

# --- 7. VISUALIZACIÓN DE PREDICCIÓN ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, pred_r, alpha=0.3, color='teal')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.title('Regresión: Predicción de días para ser tendencia')
plt.xlabel('Días Reales')
plt.ylabel('Días Predichos')
plt.tight_layout()
plt.show()