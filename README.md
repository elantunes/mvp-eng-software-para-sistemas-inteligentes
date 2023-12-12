# Instituto Moove-se de Saúde

Este é um **Sistema de Predição de Fumantes** para o **MVP** do curso de **Pós-graduação em Engenharia de Software** da **PUC-Rio**.

Este projeto fornece uma Interface Gráfica para o preenchimento de informações de uma pessoa e predizer através de Machine Learning se esta pessoa é uma possível fumante.

Informações consideradas para a predição:

idade, altura, peso, cintura, visão(esquerda), visão(direita), audição(esquerda), audição(direita), sistólica, relaxado, açucar no sangue em jejum, colesterol, triglicerídos, HDL, LDL, hemoglobina, proteína na urina, creatinina sérica, AST, ALT, Gtp, cáries dentárias e tártaro.


## Como Usar

1) Rode a API através dos comandos abaixo em eu terminal de preferência:

Crie um ambiente virtual
```powershell
python -m venv env
```

Para ativar o ambiente virtual em ambientes UNIX
```powershell
source .\env\Scripts\activate
```


Para ativar o ambiente virtual em ambientes Windows
```powershell
.\env\Scripts\Activate.ps1
```

Instale os requerimentos
```powershell
pip install -r requirements.txt
```

Execute a API
```powershell
flask run --host 0.0.0.0 --port 5000 --reload
```


2) Acesse no navegador o arquivo index.html que esta dentro da pasta FrontEnd e aproveite! :rocket: