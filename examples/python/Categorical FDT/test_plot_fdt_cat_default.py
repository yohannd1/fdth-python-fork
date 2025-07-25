import pandas as pd
from fdth import fdt  
from fdth import plot_fdt_cat  

# Dados de exemplo corrigidos
data = {
    "Category": ["Segurança"] * 18 + 
               ["Trânsito"] * 17 + 
               ["Trans. Público"] * 16 + 
               ["Saúde"] * 7 + 
               ["Educação"] * 5 + 
               ["Outros"] * 3
}

df = pd.DataFrame(data)

fdt_result = fdt(df, sort=True, decreasing=True)
fdt_df = fdt_result.fdts["Category"]

# Plotar o gráfico para um tipo específico
types = ['fb', 'fp', 'fd', 'rfb', 'rfp', 'rfd', 'rfpb', 'rfpp', 'rfpd', 
         'cfb', 'cfp', 'cfd', 'cfpb', 'cfpp', 'cfpd', 'pa']
type = 'pa'  # Escolha o tipo desejado

plot_fdt_cat(fdt_df, plot_type=type, xlab="Category", ylab="Values", main=f"Type: {type}")
