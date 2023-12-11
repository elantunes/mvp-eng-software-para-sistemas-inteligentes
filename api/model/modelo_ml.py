"""Módulo para escrever arquivo em disco."""
import pandas as pd
import pickle


class ModeloMl():
    """Classe que para manipular um arquivo de Modelo de Machine Learning"""

    def __abrir_do_disco(self, file_name:str):
        '""Abre o modelo do disco.""'
        return pickle.load(open(file_name, 'rb'))
    
    def __init__(self, file_name:str):
        """__init__"""
        self.file_name = file_name
        self.__modelo = self.__abrir_do_disco(file_name)


    def predizer(self, dados):
        """Faz a predição dos valores (0 ou 1)"""
        diagnostico = self.__modelo.predict(dados.reshape(1, -1))
        return int(diagnostico[0])


    def salvar_em_disco(self, modelo):
        """Salva o modelo em disco."""
        # Salva o modelo no disco
        pickle.dump(modelo, open(self.file_name, "wb"))
        pickle.close()
