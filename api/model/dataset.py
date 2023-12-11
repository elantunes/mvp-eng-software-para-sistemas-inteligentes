import pandas as pd

class Dataset():
    """Dataset"""

    def abrir_dataset_do_disco(file_name:str):
        '""Abre o dataset do disco.""'
        dataset = pd.read_csv(file_name, delimiter=',')
        return dataset