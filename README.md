#gesture Recognition - Modelli Classici (SVM, Random Forest, KNN)

Questa repository contiene l'implementazione di tre modelli di machine learning classici per il riconoscimento di gesture basati su dati statistici.

##struttura della repository

- `preprocessing.py` : script per il preprocessing dei dati
  - Normalizzazione con MinMaxScaler
  - Label encoding
  - Feature selection con SelectKBest
  - Salvataggio dei dati preprocessati in `preprocessed_data.pkl`

- `model.py` : script per l'addestramento dei modelli
  - Modelli implementati:
    - SVM (kernel lineare)
    - Random Forest
    - KNN
  - I modelli addestrati vengono salvati in `trained_models.pkl`

- `evaluate.py` : script per valutare i modelli sui dati di test
  - Stampa accuracy e classification report
  - Permette di confrontare le performance tra i modelli

- `evaluate_check.py` : script opzionale per controllare il corretto addestramento e preprocessing
  - Stampa forma dei dati e prime righe
  - Controlla che i modelli siano addestrati correttamente

- `preprocessed_data.pkl` : dati preprocessati salvati (generati da `preprocessing.py`)
- `trained_models.pkl` : modelli addestrati salvati (generati da `model.py`)


