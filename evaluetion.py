# evaluate.py
import pickle
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test, le, name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---", flush=True)
    print("Accuracy:", acc, flush=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_), flush=True)
    return acc

if __name__ == "__main__":
    # Carica dati preprocessati e modelli addestrati
    with open("preprocessed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test, le, selector = pickle.load(f)
    with open("trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)

    # Valuta tutti i modelli e stampa metriche
    for name, model in trained_models.items():
        evaluate_model(model, X_test, y_test, le, name=name)



