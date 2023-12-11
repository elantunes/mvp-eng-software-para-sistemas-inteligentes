"""Módulo para ignorar warnings."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flask import redirect, request
from flask_openapi3 import OpenAPI, Info, Tag
from logger import logger
from schemas import *
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model.modelo_ml import ModeloMl

warnings.filterwarnings("ignore")


info = Info(title="API do Sistema de Fumantes", version="1.0.0")
app = OpenAPI(__name__, info=info)


home_tag = Tag(name="Documentação", description="Seleção de documentação: Swagger," \
    "Redoc ou RapiDoc.")
predicoes_tag = Tag(name="Predicao", description="Verifica se o cliente é um " \
    "fumante ou não.")

################################################################################
# GET
################################################################################


@app.get('/', tags=[home_tag])
def home():
    """Redireciona para /openapi, tela que permite a escolha do estilo de documentação.
    """
    return redirect('/openapi')


################################################################################
# POST
################################################################################


@app.post('/predicao',
          tags=[predicoes_tag],
          responses={"200": PredicaoViewSchema, "404": ErrorSchema})
def verifica_predicao(form: PredicaoPostFormSchema):
    """Verifica a predição de um cliente"""
    try:

        logger.debug("Incluindo um aluguel")

        xinput = np.array([form.idade,170,60,80,.8,.8,1,1,138,
86,89,242,182,55,151,15.8,1,1,21,16,22,0,0])

        NOME_ARQUIVO_MODELO_ML = "modelos_ml/fumantes.pkl"
        modelo = ModeloMl(NOME_ARQUIVO_MODELO_ML)
        predicao = modelo.predizer(xinput)

        # session = Session()

        # veiculo = session.get(Veiculo, form.id_veiculo)

        # aluguel = Aluguel(
        #     id = None,
        #     id_cliente = form.id_cliente,
        #     id_veiculo = form.id_veiculo,
        #     data_inicio = form.data_inicio,
        #     data_termino = form.data_termino,
        #     valor = veiculo.valor_diaria * ((form.data_termino - form.data_inicio).days + 1),
        #     cliente = None,
        #     veiculo = None
        # )

        # session.add(aluguel)
        # session.commit()

        # cliente = session.get(Cliente, aluguel.id_cliente)

        # aluguel.cliente = cliente
        # aluguel.veiculo = veiculo

        logger.debug("Aluguel incluído com sucesso!")
        
        return show_aluguel(aluguel), 200

    except Exception as e:
        logger.warning(f"Erro ao adicionar um aluguel, {e}")
        return {"message": e.__traceback__}, 500

# def arredonda(valor):
#     """Arredonda um float."""
#     return round(valor,4)


# def predizer(param_dados, param_scaler, param_modelo):
#     """Faz a predição dos valores (0 ou 1)"""
#     rescaled = param_scaler.transform(param_dados.reshape(1, -1))
#     diagnostico = param_modelo.predict(rescaled)
#     return int(diagnostico[0])


print("Olá!\n")

# Carga do dataset
print("Obtendo o dataset...")


# Informa a URL de importação do dataset
URL = 'https://raw.githubusercontent.com/elantunes/mvp-eng-software-para-sistemas-inteligentes/'\
      'main/datasets/fumantes-reduzido.csv'

# Lê o arquivo
dataset = pd.read_csv(URL, delimiter=',')


# Mostra as primeiras linhas do dataset
dataset.head()
print("Dataset obtido!\n")


# Separação em conjunto de treino e conjunto de teste com holdout
TEST_SIZE = .25 # tamanho do conjunto de teste
SEED = 7 # semente aleatória
array = dataset.values
numero_colunas_dataset = len(dataset.columns)-1

X = array[:,0:numero_colunas_dataset]
y = array[:,numero_colunas_dataset]

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=TEST_SIZE, shuffle=True, random_state=SEED, stratify=y) # holdout com estratificação


# Parâmetros e partições da validação cruzada
SCORING = 'accuracy'
NUM_PARTICOES = 10
 # validação cruzada com estratificação
kfold = StratifiedKFold(n_splits=NUM_PARTICOES, shuffle=True, random_state=SEED)


# Modelagem e Inferência
np.random.seed(SEED) # definindo uma semente global


# Lista que armazenará os modelos
models = []


# Listas para armazenar os armazenar os pipelines e os resultados para todas as visões do dataset
pipelines = []
results = []
names = []


# Criando os elementos do pipeline

# Algoritmos que serão utilizados
knn = ('KNN', KNeighborsClassifier())
cart = ('CART', DecisionTreeClassifier(criterion='entropy'))
naive_bayes = ('NB', GaussianNB())
svm = ('SVM', SVC())

# Transformações que serão utilizadas
standard_scaler = ('StandardScaler', StandardScaler())
min_max_scaler = ('MinMaxScaler', MinMaxScaler())


# Montando os pipelines

# Dataset original
#pipelines.append(('KNN', Pipeline([knn])))
#pipelines.append(('CART', Pipeline([cart])))
#pipelines.append(('NB', Pipeline([naive_bayes])))
#pipelines.append(('SVM', Pipeline([svm])))

# Dataset Padronizado
#pipelines.append(('KNN-padr', Pipeline([standard_scaler, knn])))
#pipelines.append(('CART-padr', Pipeline([standard_scaler, cart])))
#pipelines.append(('NB-padr', Pipeline([standard_scaler, naive_bayes])))
#pipelines.append(('SVM-padr', Pipeline([standard_scaler, svm])))

# Dataset Normalizado
#pipelines.append(('KNN-norm', Pipeline([min_max_scaler, knn])))
#pipelines.append(('CART-norm', Pipeline([min_max_scaler, cart])))
#pipelines.append(('NB-norm', Pipeline([min_max_scaler, naive_bayes])))
#pipelines.append(('SVM-norm', Pipeline([min_max_scaler, svm])))


# Executando os pipelines
for name, model in pipelines:
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=SCORING)
    results.append(cv_results)
    names.append(name)
    msg = f'{name}\t {arredonda(cv_results.mean())}\t {arredonda(cv_results.std())}'
    #print(msg)

print('\n')

# Boxplot de comparação dos modelos
# fig = plt.figure(figsize=(25,6))
# fig.suptitle('Comparação dos Modelos - Dataset orginal, padronizado e normalizado')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names, rotation=90)
# plt.show()

# Otimização dos hiperparâmetros

# Tuning do KNN

pipelines = []

# Definindo os componentes do pipeline
cart = ('CART', DecisionTreeClassifier())
standard_scaler = ('StandardScaler', StandardScaler())
min_max_scaler = ('MinMaxScaler', MinMaxScaler())

#pipelines.append(('cart-orig', Pipeline(steps=[cart])))
#pipelines.append(('cart-padr', Pipeline(steps=[standard_scaler, cart])))
#pipelines.append(('cart-norm', Pipeline(steps=[min_max_scaler, cart])))

param_grid = { 'CART__criterion' : ['entropy'] }

# Prepara e executa o GridSearchCV
# for name, model in pipelines:
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=SCORING, cv=kfold)
#     grid.fit(X_train, y_train)
    #imprime a melhor configuração
    #print(f'Sem tratamento de missings: {name} - '\
        #f'Melhor: {grid.best_score_} usando: {grid.best_params_}')

print('\n')

# Finalização do Modelo

# CART ###########################################################################################

# Preparação do modelo de treino
#scaler = MinMaxScaler().fit(X_train) # ajuste do scaler com o conjunto de treino

# aplicação da normalização no conjunto de treino
#rescaledX = scaler.transform(X_train)

# aplicação da hiperparametrização no conjunto de treino
# modelCart = DecisionTreeClassifier(criterion='entropy')

# modelCart.fit(X_train, y_train)

NOME_ARQUIVO_MODELO_ML = "modelos_ml/fumantes.pkl"
#modelo = ModeloMl(NOME_ARQUIVO_MODELO_ML)
#modelo.salvar_em_disco(modelCart)
#modeloCart = ModeloMl().abrir_do_disco(NOME_ARQUIVO_MODELO_ML)


#.salvar_em_disco(modelCart, NOME_ARQUIVO_MODELO_ML)


# Estimativa da acurácia no conjunto de teste
#rescaledTestX = scaler.transform(X_test) # aplicação da normalização no conjunto de teste
# predictions = modelCart.predict(rescaledTestX)
# print('Estimativas do dataset de teste usando CART')
# print(f'Acurácia: {accuracy_score(y_test, predictions)}')
# print(f'Precisão: {arredonda(precision_score(y_test, predictions))}')


# Estimativa da acurácia no conjunto de TODO dataset
# rescaledX = scaler.transform(X) # aplicação da padronização no conjunto de todo dataset
# predictions = modelCart.predict(rescaledX)
# print('Estimativas do dataset completo usando CART:')
# print(f'Acurácia: {accuracy_score(y, predictions)}')
# print(f'Precisão: {arredonda(precision_score(y, predictions))}')


# SVM ############################################################################################

# Preparação do modelo de treino
# scaler = StandardScaler().fit(X_train) # ajuste do scaler com o conjunto de treino
# rescaledX = scaler.transform(X_train) # aplicação da normalização no conjunto de treino
# modelSVM = SVC()
# modelSVM.fit(rescaledX, y_train)


# Estimativa da acurácia no conjunto de teste
# rescaledTestX = scaler.transform(X_test) # aplicação da normalização no conjunto de teste
# predictions = modelSVM.predict(rescaledTestX)
# print('Estimativas do dataset de teste usando SVM:')
# print(f'Acurácia: {accuracy_score(y_test, predictions)}')
# print(f'Precisão: {arredonda(precision_score(y_test, predictions))}')


# Estimativa da acurácia no conjunto de TODO dataset
# rescaledX = scaler.transform(X) # aplicação da padronização no conjunto de todo dataset
# predictions = modelSVM.predict(rescaledX)
# print('Estimativas do dataset completo usando SVM:')
# print(f'Acurácia: {accuracy_score(y, predictions)}')
# print(f'Precisão: {arredonda(precision_score(y, predictions))}')
# print('\n')


##################################################################################################

# Simulando a aplicação do modelo em dados não vistos

# CART
#scaler = MinMaxScaler().fit(X_train) # ajuste do scaler com o conjunto de treino
#model = modelCart

modelo = ModeloMl(NOME_ARQUIVO_MODELO_ML)

X_input = np.array([27,160,60,81,.8,.6,1,1,119,
70,130,192,115,42,127,12.7,1,.6,22,19,18,0,1])
print(modelo.predizer(X_input))

X_input = np.array([69,170,60,80,.8,.8,1,1,138,
86,89,242,182,55,151,15.8,1,1,21,16,22,0,0])
print(modelo.predizer(X_input))

X_input = np.array([82,150,65,81.5,1.2,1.2,1,1,134,
86,86,238,117,63,152,12,1,0.9,19,11,16,0,0])
print(modelo.predizer(X_input))

X_input = np.array([31,160,60,86,.7,.6,1,1,133,
80,139,223,151,44,149,16.3,1,1.1,26,34,38,0,1])
print(modelo.predizer(X_input))

X_input = np.array([71,165,65,84,1,1,1,1,120,
76,95,235,132,52,166,13.7,4,.9,29,24,13,0,0])
print(modelo.predizer(X_input))

##################################################################################################

# X_input = np.array([27,160,60,81,.8,.6,1,1,119,
# 70,130,192,115,42,127,12.7,1,.6,22,19,18,0,1])
# diagnosis = predizer(X_input, scaler, model)
# print(f'#1\t {int(X_input[0])}\t Esperado:0\t Atingido:{diagnosis}')

# X_input = np.array([69,170,60,80,.8,.8,1,1,138,
# 86,89,242,182,55,151,15.8,1,1,21,16,22,0,0])
# diagnosis = predizer(X_input, scaler, model)
# print(f'#2\t {int(X_input[0])}\t Esperado:1\t Atingido:{diagnosis}')

# X_input = np.array([82,150,65,81.5,1.2,1.2,1,1,134,
# 86,86,238,117,63,152,12,1,0.9,19,11,16,0,0])
# diagnosis = predizer(X_input, scaler, model)
# print(f'#3\t {int(X_input[0])}\t Esperado:0\t Atingido:{diagnosis}')

# X_input = np.array([31,160,60,86,.7,.6,1,1,133,
# 80,139,223,151,44,149,16.3,1,1.1,26,34,38,0,1])
# diagnosis = predizer(X_input, scaler, model)
# print(f'#4\t {int(X_input[0])}\t Esperado:1\t Atingido:{diagnosis}')

# X_input = np.array([71,165,65,84,1,1,1,1,120,
# 76,95,235,132,52,166,13.7,4,.9,29,24,13,0,0])
# diagnosis = predizer(X_input, scaler, model)
# print(f'#5\t {int(X_input[0])}\t Esperado:0\t Atingido:{diagnosis}')
