import numpy as np
import pandas as pd

def make_fdt_simple(x, start, end, h, right=False):
    """
    Create a simple frequency distribution table.

    Parameters:
    x (array-like): The data array.
    start (float): The starting point of the distribution range.
    end (float): The endpoint of the distribution range.
    h (float): The class interval width.
    right (bool): Whether to include the right endpoint in each interval.

    Returns:
    DataFrame: A frequency distribution table with class limits, frequencies, relative frequencies, 
               cumulative frequencies, and cumulative percentages.
    """
    bins = np.arange(start, end + h, h)
    labels = [f"[{round(bins[i], 2)}, {round(bins[i + 1], 2)})" for i in range(len(bins) - 1)]
    freq = pd.cut(x, bins=bins, right=right, labels=labels).value_counts()
    freq = freq[freq > 0]  # Remove classes with zero frequency
    labels = freq.index.tolist()  # Atualiza os rótulos para considerar somente os existentes
    rf = freq / len(x)
    rfp = rf * 100
    cf = freq.cumsum()
    cfp = (cf / len(x)) * 100
    
    table = pd.DataFrame({
        'Class limits': labels,
        'f': freq.values,
        'rf': rf.values,
        'rf(%)': rfp.values,
        'cf': cf.values,
        'cf(%)': cfp.values
    })

    table.index = np.arange(1, len(table) + 1)
    
    return table
