import numpy as np
import pickle

from flask_openapi3 import OpenAPI, Info


info = Info(title="API do Sitema de Fumantes", version="1.0.0")
app = OpenAPI(__name__, info=info)

print("Olá2!")

ml_path = 'modelos_ml/fumantes_knn.pkl'

modelo = pickle.load(open(ml_path, 'rb'))

X_input = np.array([69,
                    170,
                    60,
                    80,
                    0.8,
                    0.8,
                    1,
                    1,
                    138,
                    86,
                    89,
                    242,
                    182,
                    55,
                    151,
                    15.8,
                    1,
                    1,
                    21,
                    16,
                    22,
                    0,
                    0
                ])


# Faremos o reshape para que o modelo entenda que estamos passando
diagnosis = modelo.predict(X_input.reshape(1, -1))

print(int(diagnosis[0]))