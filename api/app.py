import pickle
import warnings

from pickle import dump

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flask_openapi3 import OpenAPI, Info
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

warnings.filterwarnings("ignore")


info = Info(title="API do Sitema de Fumantes", version="1.0.0")
app = OpenAPI(__name__, info=info)

print("Olá!\n")

# Carga do dataset
print("Obtendo o dataset...")


# Informa a URL de importação do dataset
URL = "https://raw.githubusercontent.com/elantunes/mvp-eng-software-para-sistemas-inteligentes/mai\
n/datasets/fumantes-reduzido.csv"

# Lê o arquivo
dataset = pd.read_csv(URL, delimiter=',')


# Mostra as primeiras linhas do dataset
dataset.head()
print("Dataset obtido!")


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

# Criando os modelos e adicionando-os na lista de modelos
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))

# Listas para armazenar os resultados
# results = []
# names = []

# Avaliação dos modelos
# for name, model in models:
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

#     print(msg)

# Boxplot de comparação dos modelos
# fig = plt.figure(figsize=(15,10))
# fig.suptitle('Comparação dos Modelos')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
#plt.show()

# Criação e avaliação de modelos: dados padronizados e normalizados

# Listas para armazenar os armazenar os pipelines e os resultados para todas as visões do dataset
pipelines = []
results = []
names = []


# Criando os elementos do pipeline

# Algoritmos que serão utilizados
#knn = ('KNN', KNeighborsClassifier())
#cart = ('CART', DecisionTreeClassifier(criterion='entropy'))
#naive_bayes = ('NB', GaussianNB())
#svm = ('SVM', SVC())

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
    msg = f"{name} {cv_results.mean()} {cv_results.std()}"
    print(msg)

print('\n')

# Boxplot de comparação dos modelos
# fig = plt.figure(figsize=(25,6))
# fig.suptitle('Comparação dos Modelos - Dataset orginal, padronizado e normalizado')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names, rotation=90)
#plt.show()

# Otimização dos hiperparâmetros

# Tuning do KNN

pipelines = []

# Definindo os componentes do pipeline
cart = ('CART', DecisionTreeClassifier())
standard_scaler = ('StandardScaler', StandardScaler())
min_max_scaler = ('MinMaxScaler', MinMaxScaler())

pipelines.append(('cart-orig', Pipeline(steps=[cart])))
pipelines.append(('cart-padr', Pipeline(steps=[standard_scaler, cart])))
pipelines.append(('cart-norm', Pipeline(steps=[min_max_scaler, cart])))

param_grid = { 'CART__criterion' : ['entropy'] }

# Prepara e executa o GridSearchCV
# for name, model in pipelines:
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#     grid.fit(X_train, y_train)
    # imprime a melhor configuração
    #print("Sem tratamento de missings: %s - Melhor: %f usando\ %s" % (name, grid.best_score_, grid.best_params_))


# Finalização do Modelo

# Avaliação do modelo com o conjunto de testes

# Preparação do modelo de treino
# scaler = StandardScaler().fit(X_train) # ajuste do scaler com o conjunto de treino
# rescaledX = scaler.transform(X_train) # aplicação da padronização no conjunto de treino
# model = KNeighborsClassifier(metric='manhattan', n_neighbors=21)
# model.fit(rescaledX, y_train)

# CART ###########################################################################################

# Preparação do modelo de treino
scaler = MinMaxScaler().fit(X_train) # ajuste do scaler com o conjunto de treino
rescaledX = scaler.transform(X_train) # aplicação da normalização no conjunto de treino
 # aplicação da hiperparametrização no conjunto de treino
modelCart = DecisionTreeClassifier(criterion='entropy')
modelCart.fit(rescaledX, y_train)


# Estimativa da acurácia no conjunto de teste
# rescaledTestX = scaler.transform(X_test) # aplicação da normalização no conjunto de teste
# predictions = modelCart.predict(rescaledTestX)
# print('Estimativa da acurácia no conjunto de teste (CART)')
# print(accuracy_score(y_test, predictions))
# print('Estimativa da precisão no conjunto de teste (CART)')
# print(precision_score(y_test, predictions))

# Preparação do modelo com TODO o dataset'
# scaler = StandardScaler().fit(X) # ajuste do scaler com TODO o dataset
# rescaledX = scaler.transform(X) # aplicação da padronização com TODO o dataset
# model = KNeighborsClassifier(metric='manhattan', n_neighbors=21)
# model.fit(rescaledX, y)

#Se faz o modelo com todo ou com o treino?
# scaler = MinMaxScaler().fit(X) # ajuste do scaler com TODO o dataset
# rescaledX = scaler.transform(X) # aplicação da padronização com TODO o dataset
# #model = DecisionTreeClassifier(criterion='entropy')
# model.fit(rescaledX, y)

# Estimativa da acurácia no conjunto de TODO dataset
rescaledX = scaler.transform(X) # aplicação da padronização no conjunto de todo dataset
predictions = modelCart.predict(rescaledX)
print('Estimativa do conjunto do dataset (CART):')
print(f'Acurácia: {accuracy_score(y, predictions)}')
print(f'Precisão: {precision_score(y, predictions)}')
print('\n')


# Salva o modelo no disco
#filename = f"modelos_ml/fumantes.pkl"
#filename = f"modelos_ml/fumantes_{knn_.lower()}.pkl"
#dump(modelCart, open(filename, 'wb'))


# SVM ############################################################################################

# Preparação do modelo de treino
scaler = StandardScaler().fit(X_train) # ajuste do scaler com o conjunto de treino
rescaledX = scaler.transform(X_train) # aplicação da normalização no conjunto de treino
modelSVM = SVC()
modelSVM.fit(rescaledX, y_train)


# Estimativa da acurácia no conjunto de teste
# rescaledTestX = scaler.transform(X_test) # aplicação da normalização no conjunto de teste
# predictions = modelSVM.predict(rescaledTestX)
# print('Estimativa da acurácia no conjunto de teste (SVM)')
# print(accuracy_score(y_test, predictions))
# print('Estimativa da precisão no conjunto de teste (SVM)')
# print(precision_score(y_test, predictions))


# Estimativa da acurácia no conjunto de TODO dataset
rescaledX = scaler.transform(X) # aplicação da padronização no conjunto de todo dataset
predictions = modelSVM.predict(rescaledX)
print('Estimativa do conjunto do dataset (SVM):')
print(f'Acurácia: {accuracy_score(y, predictions)}')
print(f'Precisão: {precision_score(y, predictions)}')
print('\n')


##################################################################################################

#Simulando a aplicação do modelo em dados não vistos

# Novos dados - não sabemos a classe!
# data = {'idade': [55],
#         'altura(cm)': [155],
#         'peso(kg)': [60],
#         'cintura(cm)': [81.3],
#         'visão(esquerda)': [1.2],
#         'visão(direita)': [1],
#         'audição(esquerda)': [1],
#         'audição(direita)': [1],
#         'sistólica': [114],
#         'relaxado': [73],
#         'açucar no sangue em jejum': [94],
#         'colesterol': [215],
#         'triglicerídos': [82],
#         'HDL': [73],
#         'LDL': [126],
#         'hemoglobina': [12.9],
#         'proteína na urina': [1],
#         'creatinina sérica': [0.7],
#         'AST': [18],
#         'ALT': [19],
#         'Gtp': [27],
#         'cáries dentárias': [0],
#         'tártaro,fumante': [1],
#         }

# data = {'idade': [69],
#         'altura(cm)': [170],
#         'peso(kg)': [60],
#         'cintura(cm)': [80],
#         'visão(esquerda)': [0.8],
#         'visão(direita)': [0.8],
#         'audição(esquerda)': [1],
#         'audição(direita)': [1],
#         'sistólica': [138],
#         'relaxado': [86],
#         'açucar no sangue em jejum': [89],
#         'colesterol': [242],
#         'triglicerídos': [182],
#         'HDL': [55],
#         'LDL': [151],
#         'hemoglobina': [15.8],
#         'proteína na urina': [1],
#         'creatinina sérica': [1],
#         'AST': [21],
#         'ALT': [16],
#         'Gtp': [22],
#         'cáries dentárias': [0],
#         'tártaro': [0]
#         }


data = {'idade': [27],
        'altura(cm)': [160],
        'peso(kg)': [60],
        'cintura(cm)': [81],
        'visão(esquerda)': [0.8],
        'visão(direita)': [0.6],
        'audição(esquerda)': [1],
        'audição(direita)': [1],
        'sistólica': [119],
        'relaxado': [70],
        'açucar no sangue em jejum': [130],
        'colesterol': [192],
        'triglicerídos': [115],
        'HDL': [42],
        'LDL': [127],
        'hemoglobina': [12.7],
        'proteína na urina': [1],
        'creatinina sérica': [0.6],
        'AST': [22],
        'ALT': [19],
        'Gtp': [18],
        'cáries dentárias': [0],
        'tártaro': [1]
        }

atributos = ['idade','altura(cm)','peso(kg)','cintura(cm)','visão(esquerda)','visão(direita)',
             'audição(esquerda)','audição(direita)','sistólica',
             'relaxado','açucar no sangue em jejum',
             'colesterol','triglicerídos','HDL','LDL','hemoglobina','proteína na urina',
             'creatinina sérica','AST','ALT','Gtp','cáries dentárias','tártaro']

entrada = pd.DataFrame(data, columns=atributos)

array_entrada = entrada.values
X_entrada = array_entrada[:,0:numero_colunas_dataset].astype(float)


# Padronização nos dados de entrada usando o scaler utilizado em X
rescaledEntradaX = scaler.transform(X_entrada)
#print(rescaledEntradaX)

model = modelSVM

# Predição de classes dos dados de entrada
#saidas = model.predict(X_entrada)
saidas = model.predict(rescaledEntradaX)
#print('Saida 1')
#print(saidas)

#################################

scaler = StandardScaler().fit(X_train) # ajuste do scaler com o conjunto de treino
rescaledX = scaler.transform(X_train) # aplicação da normalização no conjunto de treino
modelSVM = SVC()
modelSVM.fit(rescaledX, y_train)

model = modelSVM

#ml_path = 'modelos_ml/fumantes.pkl'

X_input = np.array([27,160,60,81,.8,.6,1,1,119,
70,130,192,115,42,127,12.7,1,.6,22,19,18,0,1])
rescaledEntradaX = scaler.transform(X_input.reshape(1, -1))
diagnosis = model.predict(rescaledEntradaX)
print(f'#1\t {int(X_input[0])}\t Esperado:0\t Atingido:{int(diagnosis[0])}')

X_input = np.array([69,170,60,80,.8,.8,1,1,138,
86,89,242,182,55,151,15.8,1,1,21,16,22,0,0])
rescaledEntradaX = scaler.transform(X_input.reshape(1, -1))
diagnosis = model.predict(rescaledEntradaX)
print(f'#2\t {int(X_input[0])}\t Esperado:1\t Atingido:{int(diagnosis[0])}')

X_input = np.array([82,150,65,81.5,1.2,1.2,1,1,134,
86,86,238,117,63,152,12,1,0.9,19,11,16,0,0])
rescaledEntradaX = scaler.transform(X_input.reshape(1, -1))
diagnosis = model.predict(rescaledEntradaX)
print(f'#3\t {int(X_input[0])}\t Esperado:0\t Atingido:{int(diagnosis[0])}')

X_input = np.array([31,160,60,86,.7,.6,1,1,133,
80,139,223,151,44,149,16.3,1,1.1,26,34,38,0,1])
rescaledEntradaX = scaler.transform(X_input.reshape(1, -1))
diagnosis = model.predict(rescaledEntradaX)
print(f'#4\t {int(X_input[0])}\t Esperado:1\t Atingido:{int(diagnosis[0])}')

X_input = np.array([71,165,65,84,1,1,1,1,120,
76,95,235,132,52,166,13.7,4,.9,29,24,13,0,0])
rescaledEntradaX = scaler.transform(X_input.reshape(1, -1))
diagnosis = model.predict(rescaledEntradaX)
print(f'#5\t {int(X_input[0])}\t Esperado:0\t Atingido:{int(diagnosis[0])}')
