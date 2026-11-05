# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:03:41 2026

@author: flopy
"""

#%% ................... Inicialización y Carga de Datos ...................
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.signal as sig
from matplotlib.patches import Rectangle

# Cargar el archivo ECG_TP4.mat (Asegurate de tenerlo en el directorio de trabajo)
mat_data = loadmat('ECG_TP4.mat')

fs_ecg = 1000.0  # Frecuencia de muestreo en Hz
ecg_one_lead = mat_data['ecg_lead'].flatten()
qrs_detections = mat_data['qrs_detections'].flatten()

print(f"Señal de ECG cargada con éxito. Muestras totales: {len(ecg_one_lead)}")


#%% ................... Diseño de Filtros IIR (SOS) y FIR ...................
wp = 35.0  # Frecuencia de paso [Hz]
ws = 50.0  # Frecuencia de rechazo [Hz]
Gpass = 3  # Atenuación máxima en banda de paso [dB]
Gstop = 40 # Atenuación mínima en banda de rechazo [dB]

ripple = Gpass
att = Gstop

# 1. Filtros IIR utilizando output='sos' (Secciones de Segundo Orden)
sos_butt   = sig.iirdesign(wp, ws, Gpass, Gstop, ftype='butter', fs=fs_ecg, output='sos')
sos_cheby2 = sig.iirdesign(wp, ws, Gpass, Gstop, ftype='cheby2', fs=fs_ecg, output='sos')
sos_cauer  = sig.iirdesign(wp, ws, Gpass, Gstop, ftype='cauer', fs=fs_ecg, output='sos')

# 2. Filtros FIR usando el método de la ventana (estilo clase del profe)
taps_win = 131
delta_w = 5.0

b_win_boxcar = sig.firwin2(taps_win, [0.0, wp, wp + delta_w, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], window='boxcar', fs=fs_ecg)
b_win_hamming = sig.firwin2(taps_win, [0.0, wp, wp + delta_w, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], window='hamming', fs=fs_ecg)
b_win_blackman = sig.firwin2(taps_win, [0.0, wp, wp + delta_w, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], window='blackman', fs=fs_ecg)

y_min, y_max = -100, 5


#%% ................... Funciones de Respuesta en Frecuencia y Retardo ...................
def calc_freq_sos(sos):
    return sig.freqz_sos(sos, worN=8192, fs=fs_ecg)

def calc_freq_fir(bfir):
    return sig.freqz(bfir, 1, worN=8192, fs=fs_ecg)

def calc_resp(resp_freq, leg_str, fig_n_axes):
    (f, h) = resp_freq
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    fase_rad = np.unwrap(np.angle(h))
    
    # Retardo de grupo
    omega = 2 * np.pi * f  
    retardo_s = -np.gradient(fase_rad, omega)
    retardo_ms = retardo_s * 1000  
    
    umbral_db = -80
    valido = modulo_db > umbral_db
    retardo_ms_plot = np.where(valido, retardo_ms, np.nan)
    
    (fig, (ax_mod, ax_fase)) = fig_n_axes
    
    # --- Módulo ---
    ax_mod.plot(f, modulo_db, linewidth=1.2, label=leg_str)
    ax_mod.set_ylim(y_min, y_max)
    ax_mod.set_ylabel("Módulo (dB)")
    ax_mod.set_title("Respuesta en frecuencia y plantilla")
    ax_mod.grid(True)
    
    # --- Fase ---
    ax_fase.plot(f, fase_rad, label=leg_str, linewidth=1.0)
    ax_fase.axvline(wp, color="gray", linestyle=":")
    ax_fase.axvline(ws, color="gray", linestyle=":")
    ax_fase.set_xlabel("Frecuencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.grid(True)
    
    # --- Retardo de Grupo ---
    plt.figure(2, figsize=(10, 5))
    plt.plot(f, retardo_ms_plot, label=leg_str, linewidth=1.2)
    plt.axvline(wp, color="forestgreen", linestyle="--", label=f"wp = {wp} Hz")
    plt.axvline(ws, color="firebrick", linestyle="--", label=f"ws = {ws} Hz")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Retardo de grupo (ms)")
    plt.title(f"Retardo de grupo \\tau_g(f)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", fontsize=8)


#%% ................... Gráfico Conjunto de Respuestas ...................
plt.close('all')

fig_n_axes = plt.subplots(
    2, 1, sharex=True, num=1, figsize=(10, 7), constrained_layout=True
)

# Evaluamos IIR
calc_resp(calc_freq_sos(sos_butt), 'Butterworth (IIR)', fig_n_axes)

# Evaluamos FIRs
calc_resp(calc_freq_fir(b_win_boxcar), f'FIR boxcar {taps_win}', fig_n_axes)
calc_resp(calc_freq_fir(b_win_hamming), f'FIR hamming {taps_win}', fig_n_axes)
calc_resp(calc_freq_fir(b_win_blackman), f'FIR blackman {taps_win}', fig_n_axes)

# Ploteo de la plantilla sobre el módulo
(fig, (ax_mod, ax_fase)) = fig_n_axes

rect_pasante = Rectangle(
    (0, y_min), wp, -ripple - y_min,
    facecolor="red", alpha=0.12, edgecolor="red", hatch="xxx", linewidth=1.2, label='Plantilla'       
)
ax_mod.add_patch(rect_pasante)

rect_rechazo = Rectangle(
    (ws, -att), fs_ecg / 2 - ws, y_max + att,
    facecolor="red", alpha=0.12, edgecolor="red", hatch="xxx", linewidth=1.2
)
ax_mod.add_patch(rect_rechazo)

ax_mod.axvline(wp, color="gray", linestyle=":")
ax_mod.axvline(ws, color="gray", linestyle=":")

ax_mod.legend(loc="lower left", fontsize=8)
plt.show()


#%% ................... Aplicación de Filtros a la Señal (Fase Cero / Demora) ...................
# Aplicación con filtros IIR (usando sosfiltfilt para visualización limpia o sosfilt si preferís causal)
ecg_filt_butt = sig.sosfiltfilt(sos_butt, ecg_one_lead)

# Aplicación con FIR (usando filtfilt para fase cero simétrica)
ecg_fir_hamming = sig.filtfilt(b_win_hamming, 1.0, ecg_one_lead)
ecg_fir_blackman = sig.filtfilt(b_win_blackman, 1.0, ecg_one_lead)


#%% ................... Visualización de Zoom en Ventanas de Interés ...................
cant_muestras = ecg_one_lead.shape[0]
regs_interes = {(4300, 4360), (4000, 5500)}

for ii in regs_interes:
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='int')
    
    plt.figure(figsize=(10, 5))
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='Original', linewidth=1.5, alpha=0.6, color='gray')
    plt.plot(zoom_region, ecg_filt_butt[zoom_region], label='Butterworth IIR', linewidth=1.2, color='tab:blue')
    plt.plot(zoom_region, ecg_fir_hamming[zoom_region], label='FIR Hamming', linewidth=1.2, color='tab:green')
    plt.plot(zoom_region, ecg_fir_blackman[zoom_region], label='FIR Blackman', linewidth=1.2, color='tab:purple')
    
    plt.title(f'Comparación de filtros | Muestras [{ii[0]} - {ii[1]}]')
    plt.ylabel('Amplitud [mV]')
    plt.xlabel('Muestra [n]')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


#%% ................... Resultados y Demoras por Consola ...................
retardo_fir_muestras = (taps_win - 1) / 2
retardo_fir_ms = (retardo_fir_muestras / fs_ecg) * 1000

print("==================================================")
print("       RESUMEN DE FILTROS Y RETARDOS FIR          ")
print("==================================================")
print(f"Orden del FIR (Taps)   : {taps_win}")
print(f"Retardo de grupo FIR   : {retardo_fir_muestras} muestras ({retardo_fir_ms:.2f} ms)")
print(f"Ancho de banda pasante : 0 a {wp} Hz")
print(f"Banda de rechazo       : a partir de {ws} Hz")