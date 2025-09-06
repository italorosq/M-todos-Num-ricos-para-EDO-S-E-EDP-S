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
dt = 0.01

def simulacao():
    t0 = temp_init
    tf = temp_fim
    y0 = [g_init, f_init]
    T = dt
    n = int((tf - t0) / T)

    t_valores, y_valores = rk4(reator, t0, y0, T, n)
    
    g_valores = [y[0] for y in y_valores]
    f_valores = [y[1] for y in y_valores]

    return t_valores, g_valores, f_valores

tv, gv, fv = simulacao()

todos_os_dados = {
    'Tempo': tv,
    'Concentração de G': gv,
    'Concentração de F': fv
}
