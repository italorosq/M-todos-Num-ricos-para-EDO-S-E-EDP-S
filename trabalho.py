import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


def reator(t,y):

    g = y[0]
    dgdt = 13.19 * g - 13.94 * g**2
    dfdt = 1.71 * g
    return np.array([dgdt, dfdt])

def rk4(f, t0, y0, T, n):
    t_valores = [t0]
    y_valores = [y0]
    
    t = t0
    y = y0
    for i in range(n):
        k1 = T * f(t, y)
        k2 = T * f(t + 0.5 * T, y + 0.5 * k1)
        k3 = T * f(t + 0.5 * T, y + 0.5 * k2)
        k4 = T * f(t + T, y + k3)
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


for i in dt:
        T = i
        t0 = temp_init
        tf = temp_fim
        y0 = [g_init, f_init]
        n = int((tf - t0) / T)

        print(f"Simulação com dt = {T}, n = {n}")
        t_valores, y_valores = rk4(reator, t0, y0, T, n)
    
        g_valores = [y[0] for y in y_valores]
        f_valores = [y[1] for y in y_valores]
        todos_os_dados[T] = {
        'tempo': t_valores,
        'g': g_valores,
        'f': f_valores
    }
resultados = todos_os_dados[T]

nome_arquivo = f"resultados para dt = {T}.csv".replace('.', '_')
print(f"Salvando resultados em '{nome_arquivo}'...")
df = pd.DataFrame(resultados)
df.to_csv(nome_arquivo, index=False)
print("Arquivo salvo no meu diretório do codigo.")
df = pd.read_csv(nome_arquivo)

print(f"Resultados para dt = {resultados['tempo']}:")
print(f"Tempo (t): {resultados['tempo']}")
print(f"Concentração de G (g): {resultados['g']}")
print(f"Concentração de F (f): {resultados['f']}\n")






print(df)
print(f"Valor final de F: {resultados['f'][-1]:.6f}")
print(f"Valor final de G: {resultados['g'][-1]:.6f}")

plt.figure(1,figsize=(10, 6))
plt.plot(resultados['tempo'], resultados['g'], label='Concentração de G')
plt.plot(resultados['tempo'], resultados['f'], label='Concentração de F') 
plt.xlabel('Tempo')
plt.ylabel('Concentração')
plt.title('Concentração de G e F ao longo do tempo')
plt.legend()
plt.grid(True)
plt.show()  

plt.figure(2,figsize=(10, 6))
plt.plot(resultados['g'], resultados['f'], label='F vs G')
plt.xlabel('Concentração de G')
plt.ylabel('Concentração de F')
plt.title('Diagrama de Fase')
plt.legend()
plt.grid(True)
plt.show() 

plt.figure(3,figsize=(10, 6))
plt.plot(resultados['tempo'], resultados['g'], label='Concentração de G')
plt.xlabel('Tempo')
plt.ylabel('Concentração de G')
plt.title('Concentração de G ao longo do tempo')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(4,figsize=(10, 6))
plt.plot(resultados['tempo'], resultados['f'], label='Concentração de F', color='orange')
plt.xlabel('Tempo')
plt.ylabel('Concentração de F')
plt.title('Concentração de F ao longo do tempo')
plt.legend()
plt.grid(True)
plt.show()

