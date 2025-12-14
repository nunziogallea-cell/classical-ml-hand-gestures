# preprocessing.py
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_col='target', test_size=0.2, random_state=42):
    #suddivisione features e labels
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    #verifica presenza nan, valori duplicati e ditribuzione delle classi
    print("Righe duplicate totali (inclusa label):", df.duplicated().sum(), flush=True)
    print("NaN nelle feature:", X.isnull().sum().sum(), flush=True)
    print("Distribuzione classi:\n", y.value_counts(), flush=True)

    #label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    #suddivisione dei dati per l'addestramento e per il test (80% dei dati usati per l'addestramento e 20% per il training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    
    #normalizzazione dei dati
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #feature selection
    selector = SelectKBest(score_func=f_classif, k=50)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    print("Preprocessing completato con successo.", flush=True)
    return X_train_selected, X_test_selected, y_train, y_test, le, selector

if __name__ == "__main__":
    df = pd.read_csv("DYLEM-GRID_Statistic.csv")
    X_train, X_test, y_train, y_test, le, selector = preprocess_data(df)

    # Salva dati preprocessati
    with open("preprocessed_data.pkl", "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test, le, selector), f)

