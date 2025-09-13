import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

# Cargar datos
df = pd.read_csv('vgsales.csv')

# Limpieza básica
df = df.dropna(subset=['Year', 'Publisher'])
df['Year'] = df['Year'].astype(int)

# Codificar variables categóricas
le_platform = LabelEncoder()
le_genre = LabelEncoder()
le_publisher = LabelEncoder()
df['Platform_enc'] = le_platform.fit_transform(df['Platform'])
df['Genre_enc'] = le_genre.fit_transform(df['Genre'])
df['Publisher_enc'] = le_publisher.fit_transform(df['Publisher'])

# Variable de éxito: 1 si ventas globales > mediana, 0 si no
median_sales = df['Global_Sales'].median()
df['Is_Successful'] = (df['Global_Sales'] > median_sales).astype(int)


# Variables predictoras y objetivo para regresión (solo pre-lanzamiento)
regression_features = ['Platform_enc', 'Genre_enc', 'Publisher_enc', 'Year']
X = df[regression_features]
y = df['Global_Sales']

# División train/test para regresión
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: Regresión Lineal
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Modelo 2: Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)


print('--- REGRESIÓN LINEAL ---')
print(f"MSE: {mse_lr:.4f}")
print(f"R2: {r2_lr:.4f}")

print('\n--- RANDOM FOREST REGRESSOR ---')
print(f"MSE: {mse_rf:.4f}")
print(f"RMSE: {np.sqrt(mse_rf):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")
print(f"R2: {r2_rf:.4f}")

# Validación cruzada para regresión

cv_scores_reg = cross_val_score(rf, X, y, cv=5, scoring='r2')
cv_r2_mean = cv_scores_reg.mean()
cv_r2_std = cv_scores_reg.std()
print(f"CV R2 (media ± std): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

# ===================== CLASIFICACIÓN DE ÉXITO =====================

# ===================== CLASIFICACIÓN DE ÉXITO (SOLO VARIABLES PRE-LANZAMIENTO) =====================
# Usar solo variables conocidas antes del lanzamiento
prelaunch_features = ['Platform_enc', 'Genre_enc', 'Publisher_enc', 'Year']
X_clf = df[prelaunch_features]
y_clf = df['Is_Successful']

# División train/test para clasificación
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Modelo: Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_clf, y_train_clf)
y_pred_clf = clf.predict(X_test_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)


print('\n--- RANDOM FOREST CLASSIFIER (Éxito, solo pre-lanzamiento) ---')
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision_score(y_test_clf, y_pred_clf):.4f}")
print(f"Recall: {recall_score(y_test_clf, y_pred_clf):.4f}")
print(f"F1-score: {f1_score(y_test_clf, y_pred_clf):.4f}")
print("Matriz de confusión:")
print(confusion_matrix(y_test_clf, y_pred_clf))
try:
	roc_auc = roc_auc_score(y_test_clf, clf.predict_proba(X_test_clf)[:,1])
	print(f"ROC-AUC: {roc_auc:.4f}")
except Exception as e:
	print("ROC-AUC no disponible:", e)
print("\nReporte de clasificación:")
print(classification_report(y_test_clf, y_pred_clf, target_names=['No Exitoso', 'Exitoso']))

# Validación cruzada para clasificación
cv_scores_clf = cross_val_score(clf, X_clf, y_clf, cv=5, scoring='accuracy')
print(f"CV Accuracy (media ± std): {cv_scores_clf.mean():.4f} ± {cv_scores_clf.std():.4f}")
