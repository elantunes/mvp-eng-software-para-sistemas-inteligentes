from flask_openapi3 import OpenAPI, Info

info = Info(title="API do Sitema de Fumantes", version="1.0.0")
app = OpenAPI(__name__, info=info)

print("Olá")