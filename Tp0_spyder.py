# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 20:10:29 2026

@author: flopy
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Parámetros generales
fs = 1000  # Hz - frecuencia de muestreo
N = 1000  # cantidad de muestras
ts = 1 / fs  # tiempo entre muestras


# Función para generar una señal senoidal
def mi_funcion_sen(vmax=1, dc=0, ff=3, ph=0, nn=N, fs=fs):
    ts = 1 / fs  # tiempo entre cada muestra
    tt = np.arange(0, nn) * ts  # tiempo de cada muestra
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx


# Función para generar señal diente de sierra / triangular
def mi_funcion_sierra(vmax=1, dc=0, ff=3, ph=0, nn=N, fs=fs, width=1):
    ts = 1 / fs
    tt = np.arange(0, nn) * ts
    # signal.sawtooth recibe la fase en radianes
    # width=1 genera diente de sierra, width=0.5 genera onda triangular
    xx2 = dc + vmax * signal.sawtooth(2 * np.pi * ff * tt + ph, width=width)
    return tt, xx2

tt, xx2 = mi_funcion_sierra(vmax=1, dc=0, ff=3, ph=0, nn=N, fs=fs, width=1)
plt.plot(tt, xx2)
plt.xlabel("Tiempo [seg]")
plt.ylabel("Amplitud [V]")
plt.title("Funcion diente de sierra")
plt.show()

# Comienzo de mi script
vmax = 1.5  # amplitud
dc = 0  # desplazamiento vertical
ff = 3  # frecuencia variable
ph = 0  # fase
nn = N

# Prueba de la senoidal
tt, xx = mi_funcion_sen(vmax=1.5, dc=0, ff=3, ph=0, nn=N, fs=fs)
plt.plot(tt, xx)
plt.xlabel("Tiempo [seg]")
plt.ylabel("Amplitud [V]")
plt.title("Senoidal de prueba")
plt.show()

# 1. Primera frecuencia (f = 500 Hz)
plt.figure()
tt, xx = mi_funcion_sen(ff=500)
plt.plot(tt, xx, label="500 Hz")
plt.xlabel("Tiempo [seg]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()

# 2. Segunda frecuencia (f = 999 Hz) y 3. Tercera frecuencia (f = 1001 Hz)
plt.figure()
plt.subplot(1, 2, 1)
tt, xx = mi_funcion_sen(ff=999)
plt.plot(tt, xx, label="999 Hz")
plt.xlabel("Tiempo [seg]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
tt, xx = mi_funcion_sen(ff=1001)
plt.plot(tt, xx, label="1001 Hz")
plt.xlabel("Tiempo [seg]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 4. Cuarta frecuencia (f = 2001 Hz)
plt.figure()
tt, xx = mi_funcion_sen(ff=2001)
plt.plot(tt, xx, label="2001 Hz")
plt.xlabel("Tiempo [seg]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()
plt.show()


