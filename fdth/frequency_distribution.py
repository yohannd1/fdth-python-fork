#Classe FrequencyDistribution (que decide qual classe usar dependendo do tipo de dado)

from fdth.fdt_cat_data_frame import CategoricalFrequencyDistribution
from fdth.fdt_data_frame import NumericalFrequencyDistribution
import pandas as pd
import numpy as np

class FrequencyDistribution:
    def __init__(self, data):
        """
        Inicializa a classe de distribuição de frequência dependendo do tipo de dado.
        """
        # Verificando se os dados são uma lista ou um pandas Series
        if isinstance(data, list):
            data = pd.Series(data)  # Convertendo lista para pandas Series
        elif not isinstance(data, pd.Series):
            raise ValueError("Data must be a list or pandas Series.")
        
        self.data = data

        # Se for dado categórico
        if self.data.dtype == 'object' or isinstance(self.data.iloc[0], str):
            self.distribution = CategoricalFrequencyDistribution(self.data)
        else:
            # Se for dado numérico
            self.distribution = NumericalFrequencyDistribution(self.data)

    def make_fdt(self):
        return self.distribution.make_fdt()

    def mean_fdt(self):
        return self.distribution.mean_fdt()

    def var_fdt(self):
        return self.distribution.var_fdt()

    def plot_histogram(self):
        return self.distribution.plot_histogram()
