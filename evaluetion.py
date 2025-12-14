from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_test, y_test, le, name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return acc

def plot_accuracy(models_acc):
    models = list(models_acc.keys())
    acc = list(models_acc.values())
    
    plt.figure(figsize=(8,5))
    bars = plt.bar(models, acc, color=['skyblue', 'lightgreen', 'salmon'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.3f}", ha='center', va='bottom')
    plt.ylim(0.95,1.0)
    plt.ylabel("Accuracy")
    plt.title("Confronto Accuracy modelli")
    plt.show()
