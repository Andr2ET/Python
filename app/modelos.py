# app/modelos.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def ejecutar_modelos(X, y, modelos_seleccionados, metricas_seleccionadas):
    """
    Entrena y evalúa los modelos seleccionados.
    Retorna un diccionario con resultados por cada modelo.
    """
    # Dividir el dataset en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    resultados = {}
    
    for modelo in modelos_seleccionados:
        if modelo == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000)
        elif modelo == "Random Forest":
            clf = RandomForestClassifier()
        elif modelo == "SVM":
            clf = SVC(probability=True)  # Para poder calcular ROC AUC
        else:
            continue  # Si no se reconoce, omite el modelo
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None
        
        resultados[modelo] = calcular_metricas(y_test, y_pred, y_proba, metricas_seleccionadas)
    
    return resultados

def calcular_metricas(y_true, y_pred, y_proba, metricas):
    """Calcula las métricas seleccionadas."""
    resultados = {}
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
    
    return resultados
