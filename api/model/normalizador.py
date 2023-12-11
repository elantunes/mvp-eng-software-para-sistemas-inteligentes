import pickle
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler


class Normalizador():
    """Normalizador"""

    def abrir_scaler_do_disco(file_name:str):
        '""Abre o scaler do disco.""'
        return pickle.load(open(file_name, 'rb'))


    def aplicar_scaler(X_train: ndarray, scaler: MinMaxScaler):
        """Aplicação da normalização no conjunto de treino."""
        rescaledX = scaler.transform(X_train)
        return rescaledX
    

    def configurar_scaler(X_train: ndarray):
        """Ajuste do scaler com o conjunto de treino."""
        # Preparação do modelo de treino
        scaler = MinMaxScaler().fit(X_train)
        return scaler

   
    def salvar_scaler_em_disco(scaler:MinMaxScaler, file_name:str):
        """Salva o scaler em disco."""
        # Salva o modelo no disco
        f = open(file_name, 'wb')
        pickle.dump(scaler, f)
        f.close()