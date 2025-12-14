# model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    # Carica dati preprocessati
    with open("preprocessed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test, le, selector = pickle.load(f)

    models = {
        "SVM": SVC(kernel='linear', C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} addestrato con successo.", flush=True)

    # Salva modelli addestrati
    with open("trained_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)

