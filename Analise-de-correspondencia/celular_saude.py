# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:11:08 2024

@author: Matheus Henrique Dizaro Miyamoto
"""

#%% Instalando os pacotes

! pip install pandas
! pip install numpy
! pip install scipy
! pip install plotly
! pip install seaborn
! pip install matplotlib
! pip install statsmodels
! pip install prince
#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import prince
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
import plotly.express as px
#%% Primeiros tratamentos
perfil_saude_celular = pd.read_csv('Impact_of_Mobile_Phone_on_Students_Health.csv')
# Fonte: https://www.kaggle.com/datasets/innocentmfa/students-health-and-academic-performance
perfil_saude_celular = perfil_saude_celular.filter(['Names','Mobile Phone ', 'Mobile Operating System ',
       'Mobile phone use for education', 'Mobile phone activities',
       'Helpful for studying', 'Educational Apps'])
dados_saude_celular = perfil_saude_celular.drop(columns=['Names'])

tabela_1 = pd.crosstab(perfil_saude_celular['Mobile phone activities'], perfil_saude_celular['Mobile Phone '])
tabela_2 = pd.crosstab(perfil_saude_celular['Mobile phone activities'], perfil_saude_celular['Mobile Operating System '])
tabela_3 = pd.crosstab(perfil_saude_celular['Mobile phone activities'], perfil_saude_celular['Mobile phone use for education'])
tabela_4 = pd.crosstab(perfil_saude_celular['Mobile phone activities'], perfil_saude_celular['Mobile phone activities'])
tabela_5 = pd.crosstab(perfil_saude_celular['Mobile phone activities'], perfil_saude_celular['Helpful for studying'])
tabela_6 = pd.crosstab(perfil_saude_celular['Mobile phone activities'], perfil_saude_celular['Educational Apps'])
#%% Teste qui²

tab_1 = chi2_contingency(tabela_1)

print("Mobile phone activities x Mobile Phone")
print(f"estatística qui²: {round(tab_1[0], 2)}")
print(f"p-valor da estatística: {round(tab_1[1], 4)}")
print(f"graus de liberdade: {tab_1[2]}")

tab_2 = chi2_contingency(tabela_2)

print("Mobile phone activities x Mobile Operating System")
print(f"estatística qui²: {round(tab_2[0], 2)}")
print(f"p-valor da estatística: {round(tab_2[1], 4)}")
print(f"graus de liberdade: {tab_2[2]}")

tab_3 = chi2_contingency(tabela_3)

print("Mobile phone activities x Mobile phone use for education")
print(f"estatística qui²: {round(tab_3[0], 2)}")
print(f"p-valor da estatística: {round(tab_3[1], 4)}")
print(f"graus de liberdade: {tab_3[2]}")

tab_4 = chi2_contingency(tabela_4)

print("Mobile phone activities x Mobile phone activities")
print(f"estatística qui²: {round(tab_3[0], 2)}")
print(f"p-valor da estatística: {round(tab_3[1], 4)}")
print(f"graus de liberdade: {tab_3[2]}")

tab_5 = chi2_contingency(tabela_5)

print("Mobile phone activities x Helpful for studying")
print(f"estatística qui²: {round(tab_3[0], 2)}")
print(f"p-valor da estatística: {round(tab_3[1], 4)}")
print(f"graus de liberdade: {tab_3[2]}")

tab_6 = chi2_contingency(tabela_6)

print("Mobile phone activities x Educational Apps")
print(f"estatística qui²: {round(tab_3[0], 2)}")
print(f"p-valor da estatística: {round(tab_3[1], 4)}")
print(f"graus de liberdade: {tab_3[2]}")
#%%
mca = prince.MCA(n_components=3).fit(dados_saude_celular)

# Quantidade total de categorias
mca.J_

# Quantidade de variáveis na análise
mca.K_

# Quantidade de dimensões
quant_dim = mca.J_ - mca.K_
print(mca.total_inertia_/quant_dim)
# Resumo das informações
print(f"quantidade total de categorias: {mca.J_}")
print(f"quantidade de variáveis: {mca.K_}")
print(f"quantidade de dimensões: {quant_dim}")
#%% Obtendo os eigenvalues
tabela_autovalores = mca.eigenvalues_summary

print(tabela_autovalores)
print(mca.total_inertia_)
#%% Coordenadas principais das categorias das variaveis
coord_burt = mca.column_coordinates(dados_saude_celular)
print(coord_burt)

coord_padrao = mca.column_coordinates(dados_saude_celular)/np.sqrt(mca.eigenvalues_)
print(coord_padrao)

coord_obs = mca.row_coordinates(dados_saude_celular)
print(coord_obs)
#%% Plotando mapa perceptual

chart = coord_padrao.reset_index()

var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:,0])

nome_categ=[]
for col in dados_saude_celular:
    nome_categ.append(dados_saude_celular[col].sort_values(ascending=True).unique())
    categorias = pd.DataFrame(nome_categ).stack().reset_index()

chart_df_mca = pd.DataFrame({'categoria': chart['index'],
                             'obs_x': chart[0],
                             'obs_y': chart[1],
                             'obs_z': chart[2],
                             'variavel': var_chart,
                             'categoria_id': categorias[0]})

# Segundo passo: gerar o gráfico de pontos

fig = px.scatter_3d(chart_df_mca, 
                    x='obs_x', 
                    y='obs_y', 
                    z='obs_z',
                    color='variavel',
                    text=chart_df_mca.categoria_id)
fig.show()

