"""Módulo para escrever arquivo em disco."""
import pickle


class ModeloMl():
    """Classe que para manipular um arquivo de Modelo de Machine Learning"""

    def abrir_do_disco(file_name:str):
        '""Abre o modelo do disco.""'
        return pickle.load(open(file_name, 'rb'))
    

    def predizer(modelo, dados):
        """Faz a predição dos valores (0 ou 1)"""
        diagnostico = modelo.predict(dados.reshape(1, -1))
        return int(diagnostico[0])


    def salvar_em_disco(modelo, file_name:str):
        """Salva o modelo em disco."""
        # Salva o modelo no disco
        f = open(file_name, 'wb')
        pickle.dump(modelo, f)
        f.close()
