import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros generales
N = 1000
ff = 2000 
Fs = 20000
Ts = 1 / Fs

t = np.arange(N) * Ts

freqs = np.fft.fftfreq(N, Ts) #Creo vector de frecuencias (Eje x)

plt.close('all')

# Sinusoidal 2 kHz
s1 = np.sqrt(2) * np.sin(2 * np.pi * ff * t) 

X1 = np.fft.fft(s1)

fft1 = np.abs(X1) / N #Tomo modulo, me quedo con la parte real.

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t[:100]*1000, s1[:100])
plt.title('Sinusoidal 2kHz - Dominio Tiempo')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(freqs/1000, fft1)
plt.title('Sinusoidal 2kHz - FFT')
plt.xlabel('Frecuencia [kHz]')
plt.ylabel('|X(f)|')
plt.grid()

# Sinusoidal 2W (desfase pi/2)

s2 = 2 * np.sin(2 * np.pi * ff * t + np.pi/2)

X2 = np.fft.fft(s2)

fft2 = np.abs(X2) / N # De nuevo, me quedo con la parte real

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t[:100]*1000, s2[:100])
plt.title('Sinusoidal 2W desfasada - Dominio tiempo')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(freqs/1000, fft2)
plt.title('Sinusoidal 2W desfasada - FFT')
plt.xlabel('Frecuencia [kHz]')
plt.ylabel('|X(f)|')
plt.grid()

# Ruido Normal (Valor medio = 0, var=0.1)

s3 = np.random.normal(0, np.sqrt(0.1), N)

X3 = np.fft.fft(s3)
fft3 = np.abs(X3) / N

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t*1000, s3)
plt.title('Ruido Gaussiano - Dominio del Tiempo')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(freqs/1000, fft3)
plt.title('Ruido Gaussiano - FFT')
plt.xlabel('Frecuencia [kHz]')
plt.ylabel('|X(f)|')
plt.grid()

# Ruido Uniforme (Valor medio = 0, var=0.1)
# Variable aleatoria en [-a,a] sustituyendo en varianza obtengo:
a = np.sqrt(0.3)
s4 = np.random.uniform(-a, a, N)

X4 = np.fft.fft(s4)
fft4 = np.abs(X4) / N

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t*1000, s4)
plt.title('Ruido Uniforme - Dominio de tiempo')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(freqs/1000, fft4)
plt.title('Ruido Uniforme - FFT')
plt.xlabel('Frecuencia [kHz]')
plt.ylabel('|X(f)|')
plt.grid()

# Pulso Rectangular (2 kHz, 1W, 50% => duty=0.5 )

s5 = signal.square(2 * np.pi * ff * t, duty=0.5)

X5 = np.fft.fft(s5)
fft5 = np.abs(X5) / N

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t[:100]*1000, s5[:100])
plt.title('Pulso Rectangular - Dominio de tiempo')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(freqs/1000, fft5)
plt.title('Pulso Rectangular - FFT')
plt.xlabel('Frecuencia [kHz]')
plt.ylabel('|X(f)|')
plt.grid()

plt.show()