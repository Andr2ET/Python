# app/modelos.py
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

def es_clasificacion(y):
    """
    Determina si el problema es de clasificación o regresión.
    Para clasificación se espera que las etiquetas sean discretas.
    """
    return len(np.unique(y)) <= 20 and all(isinstance(val, (int, np.integer)) for val in y)

def ejecutar_modelos(X, y, modelos_seleccionados, metricas_seleccionadas):
    """
    Entrena y evalúa los modelos seleccionados.
    Retorna un diccionario con resultados por cada modelo.
    """
    # Dividir el dataset en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    es_clasif = es_clasificacion(y)
    resultados = {}
    
    for modelo in modelos_seleccionados:
        # Obtener el modelo
        clf = obtener_modelo(modelo, es_clasif)
        if clf is None:
            continue

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Si es clasificación y el modelo tiene probabilidad, obtenerla
        if es_clasif and hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:,1]
        else:
            y_proba = None
        
        # Calcular las métricas
        resultados[modelo] = calcular_metricas(y_test, y_pred, y_proba, metricas_seleccionadas, es_clasif)
    
    return resultados

def obtener_modelo(nombre, es_clasificacion):
    """
    Obtiene el modelo correspondiente según el tipo (clasificación o regresión)
    """
    # Modelos de clasificación
    clasificadores = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)  # Para ROC AUC
    }
    
    # Modelos de regresión
    regresores = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Random Forest (Regresión)": RandomForestRegressor(),
        "SVR": SVR()
    }

    if es_clasificacion:
        return clasificadores.get(nombre)
    else:
        return regresores.get(nombre)

def calcular_metricas(y_true, y_pred, y_proba, metricas, es_clasificacion):
    """Calcula las métricas seleccionadas"""
    resultados = {}
    
    if es_clasificacion:
        if "Accuracy" in metricas:
            resultados["Accuracy"] = accuracy_score(y_true, y_pred)
        if "Precision" in metricas:
            resultados["Precision"] = precision_score(y_true, y_pred, zero_division=0)
        if "Recall" in metricas:
            resultados["Recall"] = recall_score(y_true, y_pred, zero_division=0)
        if "F1" in metricas:
            resultados["F1"] = f1_score(y_true, y_pred, zero_division=0)
        if "ROC AUC" in metricas and y_proba is not None:
            resultados["ROC AUC"] = roc_auc_score(y_true, y_proba)
    else:
        if "MSE" in metricas:
            resultados["MSE"] = mean_squared_error(y_true, y_pred)
        if "RMSE" in metricas:
            resultados["RMSE"] = mean_squared_error(y_true, y_pred, squared=False)
        if "MAE" in metricas:
            resultados["MAE"] = mean_absolute_error(y_true, y_pred)
        if "R2" in metricas:
            resultados["R2"] = r2_score(y_true, y_pred)
    
    return resultados
