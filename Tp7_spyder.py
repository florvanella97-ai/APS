#%% ................... Inicialización y Carga de Datos ...................
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.signal as sig
from matplotlib.patches import Rectangle

# Cargar el archivo ecg.mat (Asegurate de tenerlo en el directorio de trabajo)
mat_data = loadmat('ECG_TP4.mat')

fs_ecg = 1000.0  # Frecuencia de muestreo en Hz
ecg_one_lead = mat_data['ecg_lead'].flatten()
qrs_pattern1 = mat_data['qrs_pattern1'].flatten()
heartbeat_pattern1 = mat_data['heartbeat_pattern1'].flatten()
heartbeat_pattern2 = mat_data['heartbeat_pattern2'].flatten()
qrs_detections = mat_data['qrs_detections'].flatten()

print(f"Señal de ECG cargada con éxito. Muestras totales: {len(ecg_one_lead)}")


#%% ................... Diseño de Filtros (Formato SOS) ...................
wp = 35.0  # Frecuencia de paso [Hz]
ws = 50.0  # Frecuencia de rechazo [Hz]
Gpass = 3  # Atenuación máxima en banda de paso [dB]
Gstop = 40 # Atenuación mínima en banda de rechazo [dB]

ripple = Gpass
att = Gstop

# Diseño utilizando output='sos' (Secciones de Segundo Orden)
sos_butt = sig.iirdesign(wp, ws, Gpass, Gstop, ftype='butter', fs=fs_ecg, output='sos')
sos_cheby2 = sig.iirdesign(wp, ws, Gpass, Gstop, ftype='cheby2', fs=fs_ecg, output='sos')
sos_cauer = sig.iirdesign(wp, ws, Gpass, Gstop, ftype='cauer', fs=fs_ecg, output='sos')

y_min, y_max = -100, 5


#%% ................... Función para Respuestas en Frecuencia y Retardo (SOS) ...................
def calc_resp(sos, leg_str, fig_n_axes):
    # Usamos freqz_sos para secciones de segundo orden
    f, h = sig.freqz_sos(sos, worN=8192, fs=fs_ecg)
    
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    fase_rad = np.unwrap(np.angle(h))
    
    # Retardo de grupo en segundos y conversión a milisegundos (ms)
    omega = 2 * np.pi * f  
    retardo_s = -np.gradient(fase_rad, omega)
    retardo_ms = retardo_s * 1000  # <--- Expresado en ms como el profe
    
    umbral_db = -80
    valido = modulo_db > umbral_db
    retardo_ms_plot = np.where(valido, retardo_ms, np.nan)
    
    (fig, (ax_mod, ax_fase)) = fig_n_axes
    
    # --- Módulo ---
    ax_mod.plot(f, modulo_db, linewidth=1.5, label=leg_str)
    ax_mod.set_ylim(y_min, y_max)
    ax_mod.set_ylabel("Módulo (dB)")
    ax_mod.set_title("Respuesta en frecuencia y plantilla")
    ax_mod.grid(True)
    
    # --- Fase ---
    ax_fase.plot(f, fase_rad, label=leg_str, linewidth=1.2)
    ax_fase.axvline(wp, color="gray", linestyle=":")
    ax_fase.axvline(ws, color="gray", linestyle=":")
    ax_fase.set_xlabel("Frecuencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.grid(True)
    
    # --- Gráfico de Retardo de Grupo en ms (Igual al de la clase) ---
    plt.figure(2, figsize=(10, 5))
    plt.plot(f, retardo_ms_plot, label=leg_str, linewidth=1.5)
    plt.axvline(wp, color="forestgreen", linestyle="--", label=f"wp = {wp} Hz")
    plt.axvline(ws, color="firebrick", linestyle="--", label=f"ws = {ws} Hz")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Retardo de grupo (ms)")
    plt.title(f"Retardo de grupo \\tau_g(f)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", fontsize=9)


#%% ................... Graficar la Respuesta de los Filtros ...................
plt.close('all')

fig_n_axes = plt.subplots(
    2, 1, sharex=True, num=1, figsize=(10, 7), constrained_layout=True
)

calc_resp(sos_butt, 'Butterworth', fig_n_axes)
calc_resp(sos_cheby2, 'Chebyshev II', fig_n_axes)
calc_resp(sos_cauer, 'Cauer (Eliptico)', fig_n_axes)

# Ploteo de la plantilla
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

ax_mod.text(wp / 2, y_max - 6, "Banda pasante\npermitida: 0 a −3 dB", ha="center", va="top", fontsize=9)
ax_mod.text((ws + fs_ecg / 2) / 2, y_max - 6, "Banda de rechazo\npermitida: ≤ −40 dB", ha="center", va="top", fontsize=9)
ax_mod.text((wp + ws) / 2, y_min + 5, "Transición", ha="center", va="bottom", fontsize=9)

ax_mod.legend(loc="lower left", fontsize=8)
plt.show()


#%% ................... Cálculo de la Respuesta al Impulso (SOS) ...................
delta_k = sig.unit_impulse(1000)
resp_imp_butt = sig.sosfiltfilt(sos_butt, delta_k)
resp_imp_cheby2 = sig.sosfiltfilt(sos_cheby2, delta_k)
resp_imp_cauer = sig.sosfiltfilt(sos_cauer, delta_k)

plt.figure(3, figsize=(10, 4))
plt.plot(resp_imp_butt, linewidth=1.25, label='Butterworth')
plt.plot(resp_imp_cheby2, linewidth=1.25, label='Chebyshev II')
plt.plot(resp_imp_cauer, linewidth=1.25, label='Cauer')
plt.legend()
plt.ylabel("Amplitud")
plt.xlabel("Muestras")
plt.title("Respuesta al impulso (Fase Cero - SOS)")
plt.grid(True)
plt.show()


#%% ................... Aplicación de Filtro a la Señal de ECG (SOS) ...................
# Usamos sosfiltfilt para fase cero en la visualización general limpia
ecg_filt_butt = sig.sosfiltfilt(sos_butt, ecg_one_lead)
ecg_filt_cheby2 = sig.sosfiltfilt(sos_cheby2, ecg_one_lead)
ecg_filt_cauer = sig.sosfiltfilt(sos_cauer, ecg_one_lead)


#%% ................... Regiones de Interés con Zoom (Estilo Figura 6 del Profe) ...................
cant_muestras = ecg_one_lead.shape[0]

regs_interes = {
    (4300, 4360),  # Ventana específica de zoom en el pico QRS (como la Figura 6)
    (4000, 5500),  # Ventana general de complejos
    (300000, 312000), 
    (720000, 744000) 
}

for ii in regs_interes:
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='int')
    
    plt.figure(figsize=(10, 5))
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='Original', linewidth=1.5, alpha=0.6, color='gray')
    plt.plot(zoom_region, ecg_filt_butt[zoom_region], label='Butterworth', linewidth=1.2, color='tab:blue')
    plt.plot(zoom_region, ecg_filt_cheby2[zoom_region], label='Chebyshev II', linewidth=1.2, color='tab:orange')
    plt.plot(zoom_region, ecg_filt_cauer[zoom_region], label='Cauer', linewidth=1.2, color='tab:purple')
    
    plt.title(f'Comparación de filtros | Muestras [{ii[0]} - {ii[1]}]')
    plt.ylabel('Amplitud [mV]')
    plt.xlabel('Muestra [n]')
    
    axes_hdl = plt.gca()
    axes_hdl.legend(loc='upper right')
    axes_hdl.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    #%% ................... Cálculo Empírico del Desfasaje (Demora) ...................

# 1. Aplicamos el filtro de forma causal (sosfilt) para que introduzca la demora real
ecg_butt_causal = sig.sosfilt(sos_butt, ecg_one_lead)

# 2. Tomamos el primer latido detectado como referencia
idx_latido = qrs_detections[0]

# Definimos una ventana chica alrededor de ese latido para buscar el pico exacto (onda R)
ventana = np.arange(idx_latido - 25, idx_latido + 25)

# Encontramos la posición exacta del pico en la señal original y en la filtrada
pico_orig = ventana[np.argmax(ecg_one_lead[ventana])]
pico_filt = ventana[np.argmax(ecg_butt_causal[ventana])]

# 3. Calculamos la demora en muestras y en milisegundos
desfasaje_muestras = pico_filt - pico_orig
desfasaje_ms = (desfasaje_muestras / fs_ecg) * 1000

print("--- RESULTADO DEL DESFASAJE (BUTTERWORTH CAUSAL) ---")
print(f"Demora en muestras: {desfasaje_muestras} muestras")
print(f"Demora en tiempo: {desfasaje_ms:.2f} ms")