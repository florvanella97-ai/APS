# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:22:10 2026

@author: Fede
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio
from scipy.interpolate import CubicSpline
import time

#%% --------------------------------Declaraciones

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg = mat_struct['ecg_lead'].flatten().astype(float)
qrs_det = mat_struct['qrs_detections'].flatten().astype(int)     # detecciones de referencia (pico R)
qrs_pat = mat_struct['qrs_pattern1'].ravel().astype(float)     # patron de QRS
hb_pat2 = mat_struct['heartbeat_pattern2'].ravel().astype(float)  # otro laido (bonus)
fs = 1000
N = len(ecg)
t = np.arange(N) / fs

#%% ------------------------------filtrado de mediana

t0= time.time()
b_med_200 = sig.medfilt(ecg,201)
b_med = sig.medfilt(b_med_200,601)
ecg_limpio = ecg - b_med
t_proc =  time.time() - t0
print(f'Tiempo de procesamiento del filtrado: {t_proc*1000:.1f} ms')

plt.close('all')

fig, ax = plt.subplots(2, 1, figsize=(12, 7))

ax[0].plot(t, ecg, lw=0.6, label='Ecg sin filtrado')
ax[0].plot(t, b_med, 'r', lw=1.5, label=r'$\hat{b}$ (mediana 200 ms → 600 ms)')
ax[0].set_title('Registro completo'); ax[0].set_xlabel('Tiempo [s]'); ax[0].set_ylabel('Amplitud [cuentas]'); ax[0].legend(loc='upper right')
ax[0].grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
ax[0].grid(which='major', linestyle='-', linewidth=0.8, color='gray')
ax[0].minorticks_on()

z = slice(int(100*fs), int(110*fs))
ax[1].plot(t[z], ecg[z], lw=0.8, label='Ecg sin filtrado')
ax[1].plot(t[z], b_med[z], 'r', lw=1.5, label=r'$\hat{b}$')
ax[1].plot(t[z], ecg_limpio[z], 'g', lw=0.8, label=r'Ecg Filtrado')
ax[1].set_title('Detalle (Region de interes)'); ax[1].set_xlabel('Tiempo [s]')
ax[1].set_ylabel('Amplitud [cuentas]'); ax[1].legend(loc='upper right')
ax[1].grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
ax[1].grid(which='major', linestyle='-', linewidth=0.8, color='gray')
ax[1].minorticks_on()
plt.tight_layout()
plt.show()

#%% --------------------------filtrado cubic spline

# plt.close('all')

n_0 = 80
m_i = qrs_det - n_0
b_spl = CubicSpline(m_i, ecg[m_i])(np.arange(N))
ecg_filtrado = ecg - b_spl


fig, ax = plt.subplots(2, 1, figsize=(12, 7))

z = slice(int(100*fs), int(110*fs))
ax[0].plot(t[z], ecg[z], lw=0.8, label='ECG sin filtrar')
ax[0].plot(t[z], b_spl[z], 'r', lw=1.5, label=r'$\hat{b}$ (spline cúbico)')
ax[0].plot(t[z], ecg_filtrado[z], 'g', lw=0.8, label=r'ECG filtrado')
m_i_z = m_i[(m_i >= z.start) & (m_i < z.stop)]
ax[0].plot(t[m_i_z], ecg[m_i_z], 'ko', label=r'puntos $\mathcal{S}$ (PQ)')
ax[0].set_title('Estimación por splines cúbicos'); ax[0].set_xlabel('Tiempo [s]'); ax[0].set_ylabel('Amplitud [cuentas]'); ax[0].legend(loc='upper right')
ax[0].grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
ax[0].grid(which='major', linestyle='-', linewidth=0.8, color='gray')
ax[0].minorticks_on()

ax[1].plot(t[z], b_med[z], label=r'$\hat{b}$ mediana')
ax[1].plot(t[z], b_spl[z], label=r'$\hat{b}$ spline')
ax[1].set_title('Comparación de ambos estimadores de la línea de base'); ax[1].set_xlabel('Tiempo [s]'); ax[1].set_ylabel('Amplitud [cuentas]'); ax[1].legend(loc='upper right')
ax[1].grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
ax[1].grid(which='major', linestyle='-', linewidth=0.8, color='gray')
ax[1].minorticks_on()
plt.tight_layout()
plt.show()

#%% ------------------- Filitro adaptado

plt.close('all')

umbral = 0.3
pat = qrs_pat - qrs_pat.mean()          # patron sin continua
L = len(pat)
pico = int(np.argmax(qrs_pat))            # muestra del pico R dentro del patron
print(f'Longitud del patrón: {L} muestras ({L/fs*1000:.0f} ms)  |  pico R en la muestra {pico}')

fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].plot(np.arange(L), qrs_pat); ax[0].axvline(pico/fs*1000, color='r', ls='--'); ax[0].set_title('Patrón qrs_pattern1'); ax[0].set_xlabel('Tiempo [ms]')
ax[0].grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
ax[0].grid(which='major', linestyle='-', linewidth=0.8, color='gray')
ax[0].minorticks_on()
ax[1].plot(np.arange(L), pat); ax[1].set_title('Patrón sin continua'); ax[1].set_xlabel('Tiempo [ms]')
ax[1].grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
ax[1].grid(which='major', linestyle='-', linewidth=0.8, color='gray')
ax[1].minorticks_on()
plt.tight_layout()
plt.show()

def filtro_adaptado(x, patron, pico):
    # Salida del filtro adaptado alineada con el pico R: y[n] es alta si un QRS tiene su pico en n.
    patron = patron - patron.mean()
    x = x - np.mean(x)
    y_full = sig.correlate(x, patron, mode='full')     # y_full[k]: patrón comenzando en k-(L-1)
    inicio = np.arange(len(y_full)) - (len(patron) - 1)
    pos = inicio + pico                                      # posición del pico R del patrón
    y = np.zeros(len(x))
    ok = (pos >= 0) & (pos < len(x))
    y[pos[ok]] = y_full[ok]
    return y

def detectar(y, umbral, t_min_ms=250):
    picos,_ = sig.find_peaks(y, height=umbral * np.max(y), distance=int(t_min_ms * fs / 1000))
    return picos

def evaluar(det, ref, tol_ms=100):
    tol = int(tol_ms * fs / 1000)
    det = np.array(det, dtype=float)
    tp = 0

    for r in ref:
        if len(det) == 0:
            break
        distancias = np.abs(det - r)
        j = np.argmin(distancias)

        if distancias[j] <= tol:
            tp += 1
            det[j] = np.inf

    se = tp / len(ref)
    vpp = tp / len(det) if len(det) > 0 else 0
    fp = len(det) - tp
    fn = len(ref) - tp
    return fn,fp,se, vpp


y_fil = filtro_adaptado(ecg_filtrado, qrs_pat, pico)   # señal ya sin línea de base
det = detectar(y_fil, umbral)
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
fn,fp,se, vpp = evaluar(det, qrs_det)

print(f"Sensibilidad = {se*100:.2f} %  |  VPP = {vpp*100:.2f} %")

z = slice(int(100*fs), int(120*fs))
ax[0].plot(t[z], y_fil[z], lw=0.8,label='ECG filtrado')
ax[0].set_title('Filtro adaptado')
ax[0].set_xlabel('Tiempo [s]')
ax[0].set_ylabel('Amplitud [cuentas]')
ax[0].legend(loc='upper right')
ax[0].grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
ax[0].grid(which='major', linestyle='-', linewidth=0.8, color='gray')
ax[0].minorticks_on()

det_z = det[(det >= z.start) & (det < z.stop)]
ref_z = qrs_det[(qrs_det >= z.start) & (qrs_det < z.stop)]

ax[1].plot(t[z], y_fil[z], lw=0.8)
ax[1].plot(t[det_z], y_fil[det_z],'rx',label='Detecciones')
ax[1].axhline(umbral * np.max(y_fil), color='r', linestyle='--', lw=0.7, label='Umbral')
ax[1].plot(t[ref_z], y_fil[ref_z],'go', markersize=4, label='Referencia')
ax[1].set_title('Detecciones')
ax[1].set_xlabel('Tiempo [s]')
ax[1].set_ylabel('Amplitud [cuentas]')
ax[1].legend(loc='upper right')
ax[1].grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
ax[1].grid(which='major', linestyle='-', linewidth=0.8, color='gray')
ax[1].minorticks_on()

plt.tight_layout()
plt.show()