from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_col='target', test_size=0.2, random_state=42):
    #suddivisione features e labels
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    #verifica presenza valori duplicati,valori,nan e sistribuzioni delle classi
    num_duplicates = df.duplicated().sum()
    print("Righe duplicate totali (inclusa label):", num_duplicates)
    print("NaN nelle feature:", X.isnull().sum().sum())
    print("Distribuzione classi:\n", y.value_counts()) 
    
    #label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    #suddivisone dei dati per l'addestramento e per il test (80% dei dati per l'addestramento,20% per il test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    
    #normalizzazione dei sample 
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #feature selection
    selector = SelectKBest(score_func=f_classif, k=50)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    return X_train_selected, X_test_selected, y_train, y_test, le, selector
