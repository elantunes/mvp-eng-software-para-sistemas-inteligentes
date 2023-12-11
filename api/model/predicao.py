"""predicao.py"""

class Predicao():
    """CLasse da Predição do modelo de Machine Learning"""
    def __init__(
        self,
        fumante: bool):
        """
        Instancia uma Prediçao

        Arguments:
            fumante: Identifica se a predição retornou como fumante
        """
        self.fumante = fumante
