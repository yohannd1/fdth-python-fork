# FIXME: analisar esses dois tipos, e colocá-los no lugar certo

## TIPO I ##


class NumericalFDT(FrequencyDistribution):
    def __init__(
        self,
        data: pd.Series | list | np.ndarray,
        k: int | None = None,
        start: float | None = None,
        end: float | None = None,
        h: float | None = None,
        breaks: NumericalBin = "Sturges",
        right: bool = False,
        na_rm: bool = False,
    ):
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        elif isinstance(data, pd.Series):
            pass
        else:
            raise TypeError("Data must be a list, a pandas.Series or an numpy.ndarray")

        data = data.dropna()
        self._data = pd.to_numeric(data, errors="coerce").dropna()

        self.n = len(self.data)
        self.k = int(
            np.ceil(1 + 3.322 * np.log10(self.n))
        )  # Regra de Sturges para determinar os intervalos
        self.h = (self.data.max() - self.data.min()) / self.k
        self.bins = np.arange(self.data.min(), self.data.max() + self.h, self.h)

        self.freq, _ = np.histogram(self.data, bins=self.bins)
        self.midpoints = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.cum_freq = np.cumsum(self.freq)

        # Tabela de frequência
        self.fdt = pd.DataFrame(
            {
                "Intervalo": list(
                    zip(np.round(self.bins[:-1], 2), np.round(self.bins[1:], 2))
                ),
                "Frequência": self.freq,
                "Frequência Acumulada": self.cum_freq,
                "Ponto Médio": np.round(self.midpoints, 2),
            }
        )

    def get_table(self) -> pd.DataFrame:
        return self.fdt

    def mean(self):
        return np.sum(self.midpoints * self.freq) / self.n

    def median(self):
        n_2 = self.n / 2
        idx = np.where(self.cum_freq >= n_2)[0][0]

        Li = self.bins[idx]  # Limite inferior da classe mediana
        Fi = self.freq[idx]  # Frequência da classe mediana
        F_acum_antes = self.cum_freq[idx - 1] if idx > 0 else 0

        median = Li + ((n_2 - F_acum_antes) * self.h) / Fi
        return median

    def mode(self):
        pass

    def var(self):
        return np.sum(((self.midpoints - self.mean()) ** 2) * self.freq) / (self.n - 1)

    def plot_histogram(self) -> None:
        plt.hist(self.data, bins=self.bins, edgecolor="black")
        plt.title("Histograma")
        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.show()


#### LIMITES NÃO APROXIMADOS ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NumericalFDT:
    def __init__(self, data):
        """
        Inicializa a classe para dados numéricos.
        """
        if not isinstance(data, (pd.Series, list)):
            raise ValueError("Dados devem ser uma lista ou uma Series do pandas.")

        self.data = pd.Series(data) if isinstance(data, list) else data
        self.data = pd.to_numeric(
            self.data, errors="coerce"
        )  # Convertendo para numérico

        self.breaks = self._define_breaks()
        self.table = self._build_fdt_table()

    def _define_breaks(self):
        """
        Define os limites dos intervalos automaticamente com base nos dados.
        """
        bins = np.histogram_bin_edges(self.data, bins="auto")
        return {"start": bins[0], "end": bins[-1], "h": bins[1] - bins[0], "bins": bins}

    def _build_fdt_table(self):
        """
        Constrói a tabela de distribuição de frequência.
        """
        breaks = self.breaks["bins"]
        freq, _ = np.histogram(self.data, bins=breaks)
        cum_freq = np.cumsum(freq)
        intervals = list(zip(breaks[:-1], breaks[1:]))

        return pd.DataFrame(
            {"Interval": intervals, "Frequency": freq, "Cumulative Frequency": cum_freq}
        )

    def make_fdt(self):
        """
        Retorna a tabela de distribuição de frequência.
        """
        return self.table

    def mean_fdt(self):
        """
        Calcula a média da distribuição de frequência.
        """
        breaks = self.breaks["bins"]
        mids = 0.5 * (breaks[:-1] + breaks[1:])
        y = self.table["Frequency"].values
        return np.sum(y * mids) / np.sum(y)

    def var_fdt(self):
        """
        Calcula a variância da distribuição de frequência.
        """
        breaks = self.breaks["bins"]
        mids = 0.5 * (breaks[:-1] + breaks[1:])
        y = self.table["Frequency"].values
        mean = self.mean_fdt()
        return np.sum((mids - mean) ** 2 * y) / (np.sum(y) - 1)

    def median_fdt(self):
        """
        Calcula a mediana da distribuição de frequência.
        """
        fdt = self.table
        n = fdt["Cumulative Frequency"].iloc[-1]
        posM = (n / 2 <= fdt["Cumulative Frequency"]).idxmax()

        breaks = self.breaks["bins"]
        liM = breaks[posM]  # limite inferior da classe mediana

        if posM - 1 < 0:
            sfaM = 0
        else:
            sfaM = fdt["Cumulative Frequency"].iloc[posM - 1]

        fM = fdt["Frequency"].iloc[posM]
        h = self.breaks["h"]

        return liM + (((n / 2) - sfaM) * h) / fM

    def plot_histogram(self):
        """
        Plota o histograma dos dados.
        """
        plt.hist(self.data, bins=self.breaks["bins"], edgecolor="black")
        plt.title("Histograma de Dados Numéricos")
        plt.xlabel("Valores")
        plt.ylabel("Frequência")
        plt.show()
