# evaluate_check.py
#def evaluate_check(X_train, X_test, y_train, y_test, le, models):
    
    #è stata implementata per verificare l'effettivo addestramento dei modelli,ovvero:
    #-viene verificata la forma delle feature dopo aver effettuato la feature selection.
    #-vengono stampati le prime 5 righe di `X_train` per controllare i dati (verificare che siano normalizzati).
    #-vengono nuovamente addestrati e testati i modelli per un ulteriore verifica.
    
    #Questa funzione è utile per assicurarsi che:
    #-la selezione delle feature è stata applicata correttamente (con il giusto numero di feature).
    -#i modelli siano addestrati sulle feature selezionate e valutati correttamente.
    
    #Parameters:
    #-X_train, X_test: dati di addestramento e test con feature selezionate.
    #-y_train, y_test: etichette di addestramento e test.
    #-le: oggetto LabelEncoder per decodificare le etichette (target).
    #-models: Dizionario dei modelli da addestrare e testare.
    
    
    #verifica la forma di X_train e X_test dopo la feature selection
    #print(f"Shape X_train (dopo feature selection): {X_train.shape}")
    #print(f"Shape X_test (dopo feature selection): {X_test.shape}")

    #mostra le prime 5 righe di X_train per un rapido controllo visivo
    #print("Prime 5 feature di X_train:")
    #print(X_train[:5])

    #addestramento e valutazione dei modelli
    #for name, model in models.items():
        #model.fit(X_train, y_train)
        #print(f"{name} addestrato correttamente? Sì")
        
        # Calcola la previsione e l'accuracy per il test
        #acc = evaluate_model(model, X_test, y_test, le, name=name)
        
        #Aggiungi l'accuracy al dizionario dei risultati
        #models_acc[name] = acc

import pickle
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test, le, name="Model"):
    """Valuta il modello e stampa le metriche"""
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Errore durante la predizione con {name}: {e}")
        return None
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---", flush=True)
    print("Accuracy:", acc, flush=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_), flush=True)
    return acc

if __name__ == "__main__":
    # Carica dati preprocessati e modelli addestrati
    with open("preprocessed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test, le, selector = pickle.load(f)
    
    print("Shape X_train (dopo feature selection):", X_train.shape)
    print("Shape X_test (dopo feature selection):", X_test.shape)
    print("Prime 5 feature di X_train:\n", X_train[:5, :5])

    with open("trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)
    
    for name, model in trained_models.items():
        # Controllo se il modello è addestrato
        trained_flag = False
        if hasattr(model, "coef_") or hasattr(model, "feature_importances_"):
            trained_flag = True
        elif hasattr(model, "n_neighbors"):  # KNN non ha coef_
            trained_flag = True

        print(f"\n{name} addestrato correttamente? {'Sì' if trained_flag else 'No'}")

        # Valuta il modello
        evaluate_model(model, X_test, y_test, le, name=name)
