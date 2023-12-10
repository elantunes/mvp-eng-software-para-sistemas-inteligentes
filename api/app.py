import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


from flask_openapi3 import OpenAPI, Info
from pickle import dump
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


info = Info(title="API do Sitema de Fumantes", version="1.0.0")
app = OpenAPI(__name__, info=info)

print("Olá!")

# CARGA DO DATASET

print("Obtendo o dataset...")

# Informa a URL de importação do dataset
url = "https://raw.githubusercontent.com/elantunes/mvp-eng-software-para-sistemas-inteligentes/main/datasets/fumantes-reduzido.csv"

# Lê o arquivo
dataset = pd.read_csv(url, delimiter=',')

# Mostra as primeiras linhas do dataset
dataset.head()

print("Dataset obtido!")

# Separação em conjunto de treino e conjunto de teste com holdout

test_size = .25 # tamanho do conjunto de teste
seed = 7 # semente aleatória

# Separação em conjuntos de treino e teste
array = dataset.values
numero_colunas_dataset = len(dataset.columns)-1

X = array[:,0:numero_colunas_dataset]
y = array[:,numero_colunas_dataset]

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=test_size, shuffle=True, random_state=seed, stratify=y) # holdout com estratificação

# Parâmetros e partições da validação cruzada
scoring = 'accuracy'
num_particoes = 10
kfold = StratifiedKFold(n_splits=num_particoes, shuffle=True, random_state=seed) # validação cruzada com estratificação

# Modelagem e Inferência

np.random.seed(seed) # definindo uma semente global

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
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) # formatando para 3 casas decimais
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
    #print("Sem tratamento de missings: %s - Melhor: %f usando %s" % (name, grid.best_score_, grid.best_params_))


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
# rescaledX = scaler.transform(X) # aplicação da padronização no conjunto de todo dataset
# predictions = modelCart.predict(rescaledX)
# print(accuracy_score(y, predictions))
# print('Estimativa da precisão no conjunto do dataset (CART)')
# print(precision_score(y, predictions))
# print('\n')

# Salva o modelo no disco
filename = f"modelos_ml/fumantes.pkl"
#filename = f"modelos_ml/fumantes_{knn_.lower()}.pkl"
dump(modelCart, open(filename, 'wb'))


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
# rescaledX = scaler.transform(X) # aplicação da padronização no conjunto de todo dataset
# predictions = modelSVM.predict(rescaledX)
# print('Estimativa da acurácia no conjunto do dataset (SVM)')
# print(accuracy_score(y, predictions))
# print('Estimativa da precisão no conjunto do dataset (SVM)')
# print(precision_score(y, predictions))

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

data = {'idade': [69],
        'altura(cm)': [170],
        'peso(kg)': [60],
        'cintura(cm)': [80],
        'visão(esquerda)': [0.8],
        'visão(direita)': [0.8],
        'audição(esquerda)': [1],
        'audição(direita)': [1],
        'sistólica': [138],
        'relaxado': [86],
        'açucar no sangue em jejum': [89],
        'colesterol': [242],
        'triglicerídos': [182],
        'HDL': [55],
        'LDL': [151],
        'hemoglobina': [15.8],
        'proteína na urina': [1],
        'creatinina sérica': [1],
        'AST': [21],
        'ALT': [16],
        'Gtp': [22],
        'cáries dentárias': [0],
        'tártaro': [0]
        }

# data = {'idade': [27],
#         'altura(cm)': [160],
#         'peso(kg)': [60],
#         'cintura(cm)': [81],
#         'visão(esquerda)': [0.8],
#         'visão(direita)': [0.6],
#         'audição(esquerda)': [1],
#         'audição(direita)': [1],
#         'sistólica': [119],
#         'relaxado': [70],
#         'açucar no sangue em jejum': [130],
#         'colesterol': [192],
#         'triglicerídos': [115],
#         'HDL': [42],
#         'LDL': [127],
#         'hemoglobina': [12.7],
#         'proteína na urina': [1],
#         'creatinina sérica': [0.6],
#         'AST': [22],
#         'ALT': [19],
#         'Gtp': [18],
#         'cáries dentárias': [0],
#         'tártaro': [1]
#         }

atributos = ['idade','altura(cm)','peso(kg)','cintura(cm)','visão(esquerda)','visão(direita)',
             'audição(esquerda)','audição(direita)','sistólica','relaxado','açucar no sangue em jejum',
             'colesterol','triglicerídos','HDL','LDL','hemoglobina','proteína na urina',
             'creatinina sérica','AST','ALT','Gtp','cáries dentárias','tártaro']

entrada = pd.DataFrame(data, columns=atributos)

array_entrada = entrada.values
X_entrada = array_entrada[:,0:numero_colunas_dataset].astype(float)

# Padronização nos dados de entrada usando o scaler utilizado em X
#rescaledEntradaX = scaler.transform(X_entrada)
#print(rescaledEntradaX)

model = modelSVM

# Predição de classes dos dados de entrada
saidas = model.predict(X_entrada)
#saidas = model.predict(rescaledEntradaX)
print('Saida 1')
print(saidas)

#################################

ml_path = 'modelos_ml/fumantes.pkl'

X_input = np.array([27,160,60,81,.8,.6,1,1,119,70,130,192,115,42,127,12.7,1,.6,22,19,18,0,1])
diagnosis = model.predict(X_input.reshape(1, -1))
print('Saida 1 (27 - Esperado:0)')
print(int(diagnosis[0]))

X_input = np.array([69,170,60,80,0.8,0.8,1,1,138,86,89,242,182,55,151,15.8,1,1,21,16,22,0,0])
# Faremos o reshape para que o modelo entenda que estamos passando
diagnosis = model.predict(X_input.reshape(1, -1))
print('Saida 2 (69 - Esperado:1)')
print(int(diagnosis[0]))

X_input = np.array([82,150,65,81.5,1.2,1.2,1,1,134,86,86,238,117,63,152,12,1,0.9,19,11,16,0,0])
diagnosis = model.predict(X_input.reshape(1, -1))
print('Saida 3 (82 - Esperado:0)')
print(int(diagnosis[0]))

X_input = np.array([31,160,60,86.0,0.7,0.6,1.0,1.0,133.0,80.0,139.0,223.0,151.0,44.0,149.0,16.3,1.0,1.1,26.0,34.0,38.0,0,1])
diagnosis = model.predict(X_input.reshape(1, -1))
print('Saida 4 (31 - Esperado:1)')
print(int(diagnosis[0]))

X_input = np.array([71,165,65,84.0,1.0,1.0,1.0,1.0,120.0,76.0,95.0,235.0,132.0,52.0,166.0,13.7,4.0,0.9,29.0,24.0,13.0,0,0])
diagnosis = model.predict(X_input.reshape(1, -1))
print('Saida 5 (71 - Esperado:0)')
print(int(diagnosis[0]))