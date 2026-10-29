# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:57:34 2026

@author: flopy
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros generales
N = 1000
Fs = 1000
Ts = 1 / Fs
df = Fs / N  # Resolución espectral (1 Hz)

t = np.arange(N) * Ts
freqs = np.fft.fftfreq(N, Ts) # Vector de frecuencias

k0_1 = N / 4
k0_2 = N / 4 + 0.25
k0_3 = N / 4 + 0.5

# Senoidales
s1 = np.sqrt(2) * np.sin(2 * np.pi * (k0_1 * df) * t)
s2 = np.sqrt(2) * np.sin(2 * np.pi * (k0_2 * df) * t)
s3 = np.sqrt(2) * np.sin(2 * np.pi * (k0_3 * df) * t)

#Cálculo de FFT 
X1 = np.fft.fft(s1)
fft1 = np.abs(X1) / N

X2 = np.fft.fft(s2)
fft2 = np.abs(X2) / N

X3 = np.fft.fft(s3)
fft3 = np.abs(X3) / N


#PSD
psd1_db = 10 * np.log10(fft1**2)

psd2_db = 10 * np.log10(fft2**2)

psd3_db = 10 * np.log10(fft3**2 )


plt.figure(figsize=(10, 5))
plt.plot(freqs[:N//2], psd1_db[:N//2], label="k0 = N/4")
plt.plot(freqs[:N//2], psd2_db[:N//2], label="k0 = N/4 + 0.25")
plt.plot(freqs[:N//2], psd3_db[:N//2], label="k0 = N/4 + 0.5")

plt.title("Desparramo Espectral")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Diagrama espectral de potencia [dB]")
plt.xlim([230, 270])
plt.grid(True)
plt.legend()
plt.show()

# Verificación de Parseval
P_t1 = np.mean(s1**2)
P_f1 = np.sum(fft1**2)

P_t2 = np.mean(s2**2)
P_f2 = np.sum(fft2**2)

P_t3 = np.mean(s3**2)
P_f3 = np.sum(fft3**2)

print(f"Potencia s1 -> Dominio de tiempo: {P_t1:.4f} W | DFT: {P_f1:.4f} W")
print(f"Potencia s2 -> Dominio de tiempo: {P_t2:.4f} W | DFT: {P_f2:.4f} W")
print(f"Potencia s3 -> Dominio de tiempo: {P_t3:.4f} W | DFT: {P_f3:.4f} W")

# Zero Padding
N_zp = 9 * N
freqs_zp = np.fft.fftfreq(N_zp, Ts)

# FFT con zero-padding 

X1_zp = np.fft.fft(s1,N_zp)
fft1_zp = np.abs(X1_zp) / N
psd1_zp_db = 10 * np.log10(fft1_zp**2 )

X2_zp = np.fft.fft(s2, N_zp)
fft2_zp = np.abs(X2_zp) / N
psd2_zp_db = 10 * np.log10(fft2_zp**2 )

X3_zp = np.fft.fft(s3,N_zp)
fft3_zp = np.abs(X3_zp) / N
psd3_zp_db = 10 * np.log10(fft3_zp**2)


plt.figure(figsize=(10, 5))
plt.plot(freqs_zp[:N_zp//2], psd1_zp_db[:N_zp//2], label="k0 = N/4")
plt.plot(freqs_zp[:N_zp//2], psd2_zp_db[:N_zp//2], label="k0 = N/4 + 0.25")
plt.plot(freqs_zp[:N_zp//2], psd3_zp_db[:N_zp//2], label="k0 = N/4 + 0.5")

plt.title("Zero Padding: Interpolación del espectro")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Diagrama espectral de potencia [dB]")
plt.xlim([230, 270])
plt.grid(True)
plt.legend()
plt.show()