# from datetime import datetime
from pydantic import BaseModel
# from typing import List

from model import Predicao


#Classes

# class AluguelGetSchema(BaseModel):
#     """ Define como uma requisição de aluguel deve ser representada.
#     """
#     id: int


class PredicaoPostFormSchema(BaseModel):
    """ Define como uma nova predição a ser verificada deve ser representado.
    """
    idade: int
    altura: int
    peso: int
    cintura: float
    visao_esquerda: float
    visao_direita: float
    audicao_esquerda: float
    audicao_direita: float
    sistolica: float
    relaxado: float
    acucar_no_sangue_em_jejum: float
    colesterol: float
    trigliceridos: float
    HDL: float
    LDL: float
    hemoglobina: float
    proteina_na_urina: float
    creatinina_serica: float
    AST: float
    ALT: float
    Gtp: float
    caries_dentarias:bool
    tartaro:bool


# class AluguelPutFormSchema(BaseModel):
#     """ Define como deve-se fornecer os dados pra atualizar um Aluguel.
#     """
#     id_veiculo: int
#     data_inicio: datetime
#     data_termino: datetime

# class AluguelPutPathSchema(BaseModel):
#     """ Define como um remoção de aluguel deve ser representada.
#     """
#     id: int


# class AluguelRequestSchema(BaseModel):
#     """ Define como um remoção de aluguel deve ser representada.
#     """
#     id: int
#     id_veiculo: int


# class AluguelDeleteSchema(BaseModel):
#     """ Define como um remoção de aluguel deve ser representada.
#     """
#     id: int


class PredicaoViewSchema(BaseModel):
    """ Define como deve ser a estrutura da predição retornada após uma requisição
    """
    fumante: bool


# class AluguelDeleteViewSchema(BaseModel):
#     """ Define como deve ser a estrutura do aluguel retornado após uma requisição
#         de remoção.
#     """
#     id: int
#     message: str


# class ListaAlugueisSchema(BaseModel):
#     """ Define como uma listagem de aluguéis será retornada.
#     """
#     alugueis:List[AluguelViewSchema]


# Defs

def show_predicao(predicao: Predicao):
    """ Retorna uma representação da Predição seguindo o schema definido em
        PredicaoViewSchema.
    """

    return {
        'fumante': predicao.fumante
    }


# def show_alugueis(alugueis: List[Aluguel]):
#     """ Retorna uma representação de uma lista de aluguéis seguindo o schema definido em
#         ListaAlugueisSchema.
#     """
#     result = []
#     for aluguel in alugueis:
#         result.append(show_aluguel(aluguel))

#     return {'alugueis': result}
