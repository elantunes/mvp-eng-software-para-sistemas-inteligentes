from model.avaliador import Avaliador
from model.dataset import Dataset
from model.modelo_ml import ModeloMl
from model.normalizador import Normalizador

avaliador = Avaliador()

# Parâmetros    
url_dados = "datasets/fumantes-golden.csv"

# Carga dos dados
dataset = Dataset.abrir_dataset_do_disco('datasets/fumantes-golden.csv')
array = dataset.values

# Separando em dados de entrada e saída
X = array[:, 0:-1]
Y = array[:, -1]

# Método para testar modelo KNN a partir do arquivo correspondente
def test_modelo_knn():
    # Importando modelo de KNN
    path = 'modelos_ml/fumantes-modelo.pkl'
    modelo = ModeloMl.abrir_do_disco(path)
    scaler = Normalizador.abrir_scaler_do_disco('scalers/fumantes-scaler.pkl')

    # Obtendo as métricas do KNN
    acuracia, recall, precisao, f1 = avaliador.avaliar(modelo, scaler, X, Y)
    
    # Testando as métricas do KNN
    # Modifique as métricas de acordo com seus requisitos
    assert acuracia >= 0.75
    assert recall >= 0.5 
    assert precisao >= 0.5 
    assert f1 >= 0.5
