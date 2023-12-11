from pandas import DataFrame
from sklearn.model_selection import train_test_split

class PreProcessador():
    """PreProcessador"""

    def processar(dataset:DataFrame):
        """Separação em conjunto de treino e conjunto de teste com holdout."""
        TEST_SIZE = .25 # tamanho do conjunto de teste
        SEED = 7 # semente aleatória
        array = dataset.values
        numero_colunas_dataset = len(dataset.columns)-1

        X = array[:,0:numero_colunas_dataset]
        y = array[:,numero_colunas_dataset]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=TEST_SIZE, shuffle=True, random_state=SEED, stratify=y) # holdout com estratificação

        return X_train, X_test, y_train, y_test
