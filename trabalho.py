import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

## Função que define o sistema de equações diferenciais

def reator(t,y,c1, c2, c3):

    g = y[0]
    dgdt = c1 * g - c2 * g**2
    dfdt = c3 * g
    return np.array([dgdt, dfdt])

## Implementação do método de Runge-Kutta de quarta ordem (RK4)

def rk4(f, t0, y0, T, n,c1, c2, c3):
    t_valores = [t0]
    y_valores = [y0]
    
    t = t0
    y = y0
    for i in range(n):
        k1 = T * f(t, y, c1, c2, c3)
        k2 = T * f(t + 0.5 * T, y + 0.5 * k1, c1, c2, c3)
        k3 = T * f(t + 0.5 * T, y + 0.5 * k2, c1, c2, c3)
        k4 = T * f(t + T, y + k3, c1, c2, c3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + T
        t_valores.append(t)
        y_valores.append(y)

    return t_valores, y_valores

## Parâmetros iniciais e simulação

temp_init = 0
temp_fim = 1
g_init = 0.03
f_init = 0
dt = [0.05, 0.025, 0.01, 0.005]
todos_os_dados = {}  # Armazena todos os dados das simulações
dados_simulacoes = {} # Armazena os dados de cada simulação

## Correndo as simulações para diferentes valores de dt e constantes

for i in range(3):

## Entrada de constantes e garantia de uso das constantes padrão na primeira simulação

    if i == 0:
        print("Usando constantes padrão do trabalho:")
        c1, c2, c3 =  13.1, 13.94, 1.71
        simulacao= 'Constantes Padrão'
    else: 
        print(f"Simulação de numero {i+1}:")
        c1 = float(input("Digite o valor da primeira constante: "))
        c2 = float(input("Digite o valor da segunda constante: "))
        c3 = float(input("Digite o valor da terceira constante: "))
        simulacao= f'Simulação {i+1}'
## Loop para diferentes valores de dt

    for i in dt:
            T = i
            t0 = temp_init
            tf = temp_fim
            y0 = [g_init, f_init]
            n = int((tf - t0) / T)


            print(f"Simulação com dt = {T}, n = {n}\n")
            t_valores, y_valores = rk4(reator, t0, y0, T, n, c1, c2, c3)

## Separando os valores de g e f
        
            g_valores = [y[0] for y in y_valores]
            f_valores = [y[1] for y in y_valores]

            todos_os_dados[T] = {
            'tempo': t_valores,
            'g': g_valores,
            'f': f_valores
        }
            resultados = todos_os_dados[T]
            df = pd.DataFrame({
                'Tempo': resultados['tempo'],
                'G': resultados['g'],
                'F': resultados['f']
            })

## Salvando os resultados em um arquivo CSV

            nome_arquivo = f"resultados para dt ={T}.csv".replace('.', '|')
            df.to_csv(nome_arquivo, index=False, sep=';' , float_format='%.6f')
            print(f"Resultados salvos em {nome_arquivo} no diretorio atual.\n")
            print(df)
            print(f'valores finais para dt = {T:.6f}:\n G = {resultados["g"][-1]:.6f}:\n F = {resultados["f"][-1]:.6f}\n')
            
    dados_simulacoes[simulacao] = todos_os_dados
    print("Simulação concluída para todos os passos de tempo.")

sns.set_theme(style="whitegrid", palette="viridis")

