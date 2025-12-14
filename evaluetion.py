# evaluate.py
import pickle
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, le, name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---", flush=True)
    print("Accuracy:", acc, flush=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_), flush=True)
    return acc

def plot_accuracy(models_acc):
    models = list(models_acc.keys())
    acc = list(models_acc.values())

    plt.figure(figsize=(8,5))
    bars = plt.bar(models, acc, color=['skyblue', 'lightgreen', 'salmon'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.3f}", ha='center', va='bottom')
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Confronto Accuracy modelli")
    plt.show()

if __name__ == "__main__":
    # Carica dati preprocessati e modelli addestrati
    with open("preprocessed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test, le, selector = pickle.load(f)
    with open("trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)

    models_acc = {}
    for name, model in trained_models.items():
        acc = evaluate_model(model, X_test, y_test, le, name=name)
        models_acc[name] = acc

    plot_accuracy(models_acc)

