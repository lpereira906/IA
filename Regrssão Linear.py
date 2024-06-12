from matplotlib import pyplot as plt #Biblioteca de Gráficos
import pandas as pd #Biblioteca Data Frame
import pylab as plt
import numpy as np
from sklearn import linear_model #Biblioteca de APrendizado de Máquina (Modelo de Regressão Linear)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error #Importar Métricas de Erro
from sklearn.model_selection import train_test_split 
from math import sqrt


# Cria um dataset chamado 'df' que receberá os dados do csv
df = pd.read_csv("FuelConsumptionCo2.csv")

#EXIBE A ESTRUTURA DO DATAFRAME
print(df.head())

#EXIBE RESUMO DATA SET, DESVIO PADRÃO MEDIA, PERCENTIL ETC....
print(df.describe()) 

#SELECIONA AS FEATURES DO MOTOR E DO CO2, AS VARIAVEIS A SEREM USADAS

motores =  df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]
print(motores.head())

#QUEBRAR O DATA SET EM DOIS, UMA PARTE DE TREINO E OUTRA DE TESTE, UMA MOTOR TREINO E UM MOTOR TESTE (2 DATA FRAME)
#UM CO2 TREINO E UM CO2 TESTE (2 DATA FRAMES) TOTAL DE 4 DATA FRAMES CRIADOS, VARIÁVEIS QUE VAO ALIMENTAR MOTORES E CO2
# TESTE SIZE 20%, OU SEJA VAMOS RESERVAR 20% PARA TESTAR O MODELO

motores_treino, motores_test, co2_treino, co2_teste = train_test_split(motores, co2, test_size=0.2, random_state=42)
print(type(motores_treino))

#PLOTAR OS GRÁFICOS DE TREINO PARA VER A CORRELACAO
plt.scatter(motores_treino, co2_treino, color='blue')
plt.xlabel("Motor")
plt.ylabel("Emissão de CO2")
plt.show()
     

#TREINAR O ALGORITMO PARA QUE ELE ME DE A FUNCAO DO MODELO DE PREVISÃO, ELE VAI MINIMIZAR A SOMA DOS RESIDUOS/ERROS
#AO QUADRADO (JA QUE POSSO TER DIFERENCAS A BAIXO DA LINHA BASE QUE TENHAM VALORES NEGATIVOS)


# CRIAR UM MODELO DE TIPO DE REGRESSÃO LINEAR
modelo =  linear_model.LinearRegression()

# TREINAR O MODELO USANDO O DATASET DE TESTE
# PARA ENCONTRAR O VALOR DE A E B (Y = A + B.X), É O TREINAMENTO MESMO

modelo.fit(motores_treino, co2_treino)

#PRINTAR O COEFICIENTE ANGULAR E LINEAR DA RETA DE REGRESSÃO
print('(A) Intercepto: ', modelo.intercept_)
print('(B) Inclinação: ', modelo.coef_)

#PRINTAR REGRA DE REGRSSÃO LINEARplt.scatter(motores_treino, co2_treino, color='blue')
plt.plot(motores_treino, modelo.coef_[0][0]*motores_treino + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.show()

#EXECUTAR O MODELO NO DATA SET DE TESTE
# Primeiro a gente tem que fazer as predições usando o modelo e base de teste
predicoesCo2 = modelo.predict(motores_test)

#PLOTAR RETA DE REGRESSÃO NO DATA SET DE TESTE

plt.scatter(motores_test, co2_teste, color='blue')
plt.plot(motores_test, modelo.coef_[0][0]*motores_test + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.show()

#AVALIAÇÃO DO MODELO COM OS INDICES DE ERRO

#Agora é mostrar as métricas
print("Soma dos Erros ao Quadrado (SSE): %2.f " % np.sum((predicoesCo2 - co2_teste)**2))
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(co2_teste, predicoesCo2))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(co2_teste, predicoesCo2))
print ("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(co2_teste, predicoesCo2)))
print("R2-score: %.2f" % r2_score(predicoesCo2 , co2_teste) )