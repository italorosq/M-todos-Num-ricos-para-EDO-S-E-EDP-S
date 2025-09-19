import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns



def reator(t,y,c1, c2, c3):

    g = y[0]
    dgdt = cont1* g - cont2 * g**2
    dfdt = cont3 * g
    return np.array([dgdt, dfdt])

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

temp_init = 0
temp_fim = 1
g_init = 0.03
f_init = 0
dt = [0.05, 0.025, 0.01, 0.005]
todos_os_dados = {}

for i in range(3):
    if i == 0:
        print("Usando constantes padrão do trabalho:")
        cont1, cont2, cont3 =  13.1, 13.94, 1.71
    else: 
        cont1 = float(input("Digite o valor da primeira constante: "))
        cont2 = float(input("Digite o valor da segunda constante: "))
        cont3 = float(input("Digite o valor da terceira constante: "))


    for i in dt:
            T = i
            t0 = temp_init
            tf = temp_fim
            y0 = [g_init, f_init]
            n = int((tf - t0) / T)


            print(f"Simulação com dt = {T}, n = {n}\n")
            t_valores, y_valores = rk4(reator, t0, y0, T, n, cont1, cont2, cont3)
        
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

            nome_arquivo = f"resultados para dt ={T}.csv".replace('.', ',')
            df.to_csv(nome_arquivo, index=False, sep=';' , float_format='%.6f')
            print(f"Resultados salvos em {nome_arquivo} no diretorio atual.\n")
            print(df)
            print(f'valores finais para dt = {T:.6f}:\n G = {resultados["g"][-1]:.6f}:\n F = {resultados["f"][-1]:.6f}\n')
            


    print("Simulação concluída para todos os passos de tempo.")

    sns.set_theme(style="whitegrid", palette="viridis")

for dt in T:
    dados = todos_os_dados[dt]
    sns.lineplot(x=dados['tempo'], y=dados['g'], label=f'dt={dt}')
    plt.title('Concentração de G ao longo do tempo')
    plt.xlabel('Tempo')
    plt.ylabel('Concentração de G')
    plt.legend()
    plt.show()


