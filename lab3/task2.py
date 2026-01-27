import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns


def eda(df):
    print("Shape:", df.shape)
    print("Memory usage (bytes):", df.memory_usage(deep=True).sum())
    print("\nNumeric summary:")
    print(df.describe().T)
    print("\nMissing values per column:")
    print(df.isna().sum())

def prepare_data(df):
    df = df.copy()
    # quality >= 6 это 1, иначе 0
    df['target'] = (df['quality'] >= 6).astype(int)
    X = df.drop(columns=['quality', 'target', 'Id'])
    y = df['target']
    numeric_feats = X.columns.tolist()
    return X, y, numeric_feats

def build_pipelines():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    clf_knn = Pipeline([
        ("pre", numeric_transformer),
        ("clf", KNeighborsClassifier())
    ])
    clf_log = Pipeline([
        ("pre", numeric_transformer),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    clf_svm = Pipeline([
        ("pre", numeric_transformer),
        ("clf", SVC(probability=True))
    ])
    return clf_knn, clf_log, clf_svm

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm, "roc_auc": roc, "report": classification_report(y_test, y_pred, zero_division=0)}

def main():
    df = pd.read_csv("WineQT.csv") 
    eda(df)
    X, y, numeric_feats = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn_pipe, log_pipe, svm_pipe = build_pipelines()
    knn_pipe.fit(X_train, y_train)
    log_pipe.fit(X_train, y_train)
    svm_pipe.fit(X_train, y_train)

    res_knn = evaluate(knn_pipe, X_test, y_test)
    res_log = evaluate(log_pipe, X_test, y_test)
    res_svm = evaluate(svm_pipe, X_test, y_test)

    for name, res in [("KNN", res_knn), ("Logistic Regression", res_log), ("SVM", res_svm)]:
        print(f"\n{name} results:")
        for k, v in res.items():
            if k == "confusion_matrix":
                print(k, "\n", v)
            elif k == "report":
                print(v)
            else:
                print(k, ":", v)

    
    models = [knn_pipe, log_pipe, svm_pipe]
    labels = ["KNN", "Logistic Regression", "SVM"]

    plt.figure(figsize=(8,6))
    for model, label in zip(models, labels):
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые для моделей Wine")
    plt.legend(loc="lower right")
    plt.show()


    def plot_cm(cm, title):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    plot_cm(res_knn["confusion_matrix"], "KNN Confusion Matrix")
    plot_cm(res_log["confusion_matrix"], "Logistic Regression Confusion Matrix")
    plot_cm(res_svm["confusion_matrix"], "SVM Confusion Matrix")

if __name__ == "__main__":
    main()
