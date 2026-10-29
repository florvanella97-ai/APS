# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:47:22 2026

@author: flopy
"""
# Senoidal con ruido.
#2pi/N = resolucion espectral. 
#La senoidal no tiene una unica frecuencia. 

import numpy as np
import matplotlib.pyplot as plt
import math 
from matplotlib.pyplot import subplot
import scipy.stats as stats
from scipy.stats import kstest
from scipy import signal
import pandas as pd

# 1. Parámetros del experimento
N=1000        # Muestras por señal (tiempo)
R = 200         # Realizaciones (experimentos)
Omega_0 = np.pi / 2
a0 = np.sqrt(2) # Amplitud calibrada para Ps = 1 W

fs = 1000   # Hz - frecuencia de muestreo 
ts = 1/fs  

# Función para generar una señal senoidal
def mi_funcion_sen(vmax= math.sqrt(2), dc=0, ff=3, ph=0, nn=N, fs=fs):
    ts = 1 / fs                     # tiempo entre cada muestra
    tt = np.arange(0, nn) * ts      # tiempo de cada muestra
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return(tt, xx)

# Generación de la senoidal pura (1W)
vmax = math.sqrt(2)      # amplitud
dc = 0          # desplazamiento vertical
ff = 3       # frecuencia variable
ph = 0          # fase
nn = N 

tt = np.arange(0, nn) * ts
xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)

px = np.var(xx)
print("Potencia de la señal (Px):", px)


# Definimos los dos SNR a analizar
snr1_db = 3
snr2_db = 10

# Calculamos sigma para SNR = 3 dB
P_ruido1 = px / (10 ** (snr1_db / 10))
sigma1 = np.sqrt(P_ruido1)

# Calculamos sigma para SNR = 10 dB
P_ruido2 = px / (10 ** (snr2_db / 10))
sigma2 = np.sqrt(P_ruido2)

# Calcular la potencia requerida para el ruido
#P_ruido = px / snr_lineal

#sigma = np.sqrt(P_ruido)

# Vector de M frecuencias aleatorias (distribucion fr ~ u(-2,2)) 
fr = np.random.uniform(-2, 2, size=(R, 1)) #Columna de valores de frecuencias R valores
                      
Omega1 = Omega_0 + (fr * 2 * np.pi / N)

# Vectores de tiempo y matriz de ruido
n = np.arange(N)  # Vector de 0 a N-1 (fila de tiempo)

na1 = np.random.normal(0, sigma1, size=(R, N)) #ruido 3dB
na2 = np.random.normal(0, sigma2, size=(R, N)) #ruido 10dB

# Matriz de señales x(R, N) de tamaño 200 x 1000
x1 = a0 * np.sin(Omega1 * n) + na1
x2 = a0 * np.sin(Omega1 * n) + na2

#Aplicacion para 3dB usando x1

# Aplicación de zero padding
N_zp = 9 * N
freqs_zp = np.fft.fftfreq(N_zp, ts)

# FFT con zero-padding sobre la matriz x1
X1_zp = np.fft.fft(x1, N_zp, axis=1)
fft1_zp = np.abs(X1_zp) / N
psd1_zp_db = 10 * np.log10(fft1_zp**2)

# Ventana flattop
w_flat = signal.windows.flattop(N)

X2_flat = np.fft.fft(x1 * w_flat, N_zp, axis=1)
fft_flat = np.abs(X2_flat) / np.sum(w_flat)

# Ventana Blackman-Harris
w_black = signal.windows.blackmanharris(N)

X3_black = np.fft.fft(x1 * w_black, N_zp, axis=1)
fft_black = np.abs(X3_black) / np.sum(w_black)

#Ventana propuesta: Hann
w_hann = signal.windows.hann(N)

X4_hann = np.fft.fft(x1 * w_hann, N_zp, axis=1)
fft_hann = np.abs(X4_hann) / np.sum(w_hann)

#Estimadores

# Nos quedamos con la mitad positiva del espectro(simetria)
mitad = N_zp // 2
freqs_pos = freqs_zp[:mitad]

# Pasamos esas frecuencias a rad/muestra
omegas_rad = 2 * np.pi * freqs_pos * ts  

# Buscamos el índice de la frecuencia más cercana a pi/2 (a0)
#Columna donde voy a medir amplitud para todas las realizaciones (posicion)
indice_omega = np.argmin(np.abs(omegas_rad - Omega_0))

# Estimación de Amplitud (A) en Omega_0 = pi/2
# Multiplicamos por 2 para considerar la potencia total del espectro 

A_rect = 2 * fft1_zp[:, indice_omega]
A_flat = 2 * fft_flat[:, indice_omega]
A_black = 2 * fft_black[:, indice_omega]
A_hann = 2 * fft_hann[:, indice_omega]

# Estimación de Frecuencia (Frec) en el pico máximo
# Buscamos la columna del máximo en la primera mitad para cada realización
Frec_rect = np.argmax(fft1_zp[:, :mitad], axis=1)
Frec_flat = np.argmax(fft_flat[:, :mitad], axis=1)
Frec_black = np.argmax(fft_black[:, :mitad], axis=1)
Frec_hann = np.argmax(fft_hann[:, :mitad], axis=1)

# Traducimos esos índices a frecuencia en rad/muestra
W_rect = omegas_rad[Frec_rect]
W_flat = omegas_rad[Frec_flat]
W_black = omegas_rad[Frec_black]
W_hann = omegas_rad[Frec_hann]

#Proceso para SNR = 10 dB (usando x2)

X1_zp_2 = np.fft.fft(x2, N_zp, axis=1)
fft1_zp_2 = np.abs(X1_zp_2) / N

X2_flat_2 = np.fft.fft(x2 * w_flat, N_zp, axis=1)
fft_flat_2 = np.abs(X2_flat_2) / np.sum(w_flat)

X3_black_2 = np.fft.fft(x2 * w_black, N_zp, axis=1)
fft_black_2 = np.abs(X3_black_2) / np.sum(w_black)

X4_hann_2 = np.fft.fft(x2 * w_hann, N_zp, axis=1)
fft_hann_2 = np.abs(X4_hann_2) / np.sum(w_hann)

# Amplitud en Omega_0 para SNR = 10 dB
A_rect_2 = 2 * fft1_zp_2[:, indice_omega]
A_flat_2 = 2 * fft_flat_2[:, indice_omega]
A_black_2 = 2 * fft_black_2[:, indice_omega]
A_hann_2 = 2 * fft_hann_2[:, indice_omega]

# Estimación de Frecuencia (Frec) en el pico máximo para SNR = 10 dB
Frec_rect_2 = np.argmax(fft1_zp_2[:, :mitad], axis=1)
Frec_flat_2 = np.argmax(fft_flat_2[:, :mitad], axis=1)
Frec_black_2 = np.argmax(fft_black_2[:, :mitad], axis=1)
Frec_hann_2 = np.argmax(fft_hann_2[:, :mitad], axis=1)

# Traducimos esos índices a frecuencia en rad/muestra
W_rect_2 = omegas_rad[Frec_rect_2]
W_flat_2 = omegas_rad[Frec_flat_2]
W_black_2 = omegas_rad[Frec_black_2]
W_hann_2 = omegas_rad[Frec_hann_2]

# PUNTO 1: Histograma comparativo de estimación de amplitud 

#SNR = 3 dB

plt.figure(figsize=(10, 6))

# Histogramas solapados
plt.hist(A_rect, bins=20, alpha=0.4, label='Rectangular (Zero Padding)', color='blue')
plt.hist(A_flat, bins=20, alpha=0.4, label='Flat-top', color='orange')
plt.hist(A_black, bins=20, alpha=0.4, label='Blackman-Harris', color='green')
plt.hist(A_hann, bins=20, alpha=0.4, label='Hann', color='red')

# Línea de referencia del valor real a0 = sqrt(2)
plt.axvline(a0, color='black', linestyle='--', linewidth=2, label=f'Valor Real a0 ({a0:.3f})')

plt.title('Punto 1: Histograma de Estimación de Amplitud (SNR = 3 dB)')
plt.xlabel('Amplitud estimada (A)')
plt.ylabel('Frecuencia de ocurrencia (W)')
plt.legend()
plt.grid(True)
plt.show()


# Para SNR = 10 dB
plt.figure(figsize=(10, 5))
plt.hist(A_rect_2, bins=20, alpha=0.4, label='Rectangular (Zero Padding)', color='blue')
plt.hist(A_flat_2, bins=20, alpha=0.4, label='Flat-top', color='orange')
plt.hist(A_black_2, bins=20, alpha=0.4, label='Blackman-Harris', color='green')
plt.hist(A_hann_2, bins=20, alpha=0.4, label='Hann', color='red')

plt.axvline(a0, color='black', linestyle='--', linewidth=2, label=f'Valor Real a0 ({a0:.3f})')
plt.title('Punto 1: Histograma de Estimación de Amplitud en Ω0 (SNR = 10 dB)')
plt.xlabel('Amplitud estimada (A)')
plt.ylabel('Frecuencia de ocurrencia(W)')
plt.legend()
plt.grid(True)
plt.show()

#Punto 2: sesgo y varianza
# Convertimos Omega1 a vector 1D (200,) para operar sin problemas de forma
Omega1_vec = Omega1.flatten()
# SNR = 3 dB
# Amplitud
Sa_3db = [np.mean(A_rect) - a0, np.mean(A_flat) - a0, np.mean(A_black) - a0, np.mean(A_hann) - a0]
Va_3db = [np.var(A_rect), np.var(A_flat), np.var(A_black), np.var(A_hann)]

# Frecuencia
err_W_rect_3db  = W_rect - Omega1_vec
err_W_flat_3db  = W_flat - Omega1_vec
err_W_black_3db = W_black - Omega1_vec
err_W_hann_3db  = W_hann - Omega1_vec

Sw_3db = [np.mean(err_W_rect_3db), np.mean(err_W_flat_3db), np.mean(err_W_black_3db), np.mean(err_W_hann_3db)]
Vw_3db = [np.var(err_W_rect_3db), np.var(err_W_flat_3db), np.var(err_W_black_3db), np.var(err_W_hann_3db)]

# SNR = 10 dB
# Amplitud
Sa_10db = [np.mean(A_rect_2) - a0, np.mean(A_flat_2) - a0, np.mean(A_black_2) - a0, np.mean(A_hann_2) - a0]
Va_10db = [np.var(A_rect_2), np.var(A_flat_2), np.var(A_black_2), np.var(A_hann_2)]

# Frecuencia
err_W_rect_10db  = W_rect_2 - Omega1_vec
err_W_flat_10db  = W_flat_2 - Omega1_vec
err_W_black_10db = W_black_2 - Omega1_vec
err_W_hann_10db  = W_hann_2 - Omega1_vec

Sw_10db = [np.mean(err_W_rect_10db), np.mean(err_W_flat_10db), np.mean(err_W_black_10db), np.mean(err_W_hann_10db)]
Vw_10db = [np.var(err_W_rect_10db), np.var(err_W_flat_10db), np.var(err_W_black_10db), np.var(err_W_hann_10db)]

print("3dB")
print("Sa (Sesgo Amplitud):   ", Sa_3db)
print("Va (Varianza Amplitud):", Va_3db)
print("Sw (Sesgo Frecuencia): ", Sw_3db)
print("Vw (Varianza Frecuencia):", Vw_3db)

print("\n10 dB")
print("Sa (Sesgo Amplitud):   ", Sa_10db)
print("Va (Varianza Amplitud):", Va_10db)
print("Sw (Sesgo Frecuencia): ", Sw_10db)
print("Vw (Varianza Frecuencia):", Vw_10db)