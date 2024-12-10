import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import configparser

# Función para cargar configuraciones
def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return {key: float(value) for key, value in config['DEFAULT'].items()}

# Función para crear un filtro Butterworth
def butter_lowpass(cutoff, fs, order=5):
    """
    Crea un filtro pasa bajas Butterworth.
    :param cutoff: Frecuencia de corte (Hz).
    :param fs: Frecuencia de muestreo (Hz).
    :param order: Orden del filtro.
    :return: Coeficientes del filtro.
    """
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    return b, a

# Algoritmo LMS (filtro adaptativo)
def lms_filter(desired, x, mu, filter_order):
    N = len(x)
    w = np.zeros(filter_order)  # Coeficientes iniciales del filtro
    y = np.zeros(N)  # Salida del filtro
    e = np.zeros(N)  # Error
    
    for n in range(filter_order, N):
        x_n = x[n-filter_order:n][::-1]  # Ventana de entrada
        y[n] = np.dot(w, x_n)  # Salida del filtro
        e[n] = desired[n] - y[n]  # Error
        w += 2 * mu * e[n] * x_n  # Actualización de coeficientes
    return y, e, w

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Aplica un filtro pasa bajas Butterworth a los datos.
    :param data: Señal de entrada.
    :param cutoff: Frecuencia de corte (Hz).
    :param fs: Frecuencia de muestreo (Hz).
    :param order: Orden del filtro.
    :return: Señal filtrada.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Lista de archivos config disponibles
config_files = {
    1: "config/1_base.config",
    2: "config/2_frec_min_ampl_media.config",
    3: "config/3_frec_max_ampl_media.config",
    4: "config/4_frec_media_ampl_min.config",
    5: "config/5_frec_media_ampl_max.config",
    6: "config/6_frecYampl_media_pert_min.config",
    7: "config/7_frecYampl_media_pert_max.config",
    8: "config/8_error_frec_max.config",
    9: "config/9_error_ampl_max.config",
    10: "config/10_error_saturacion_medicion.config",
    11: "config/11_error_fs_baja.config"
}

# Mostrar opciones al usuario
print("Seleccione un caso de uso:")
for number, config_name in config_files.items():
    print(f"{number}. {config_name}")

# Selección del usuario
selected_case = int(input("Ingrese el número del caso de uso: "))
if selected_case not in config_files:
    raise ValueError("El número seleccionado no corresponde a un caso válido.")

# Cargar configuración seleccionada
config_path = config_files[selected_case]
config = load_config(config_path)

# Parámetros de simulación
fs = config.get('freq_sampling', 0)  # Frecuencia de muestreo
duration = config.get('simu_time', 0)  # Duración en segundos
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Configuración de ruido y perturbación
freq_noise = config.get('freq_noise', 0)
amplitude_noise = config.get('amplitude_noise', 0)
freq_disturbance = config.get('freq_disturbance', 0)
amplitude_disturbance = config.get('amplitude_disturbance', 0)

# Generar ruido y perturbaciones
noise = amplitude_noise * np.sin(2 * np.pi * freq_noise * t)
disturbance = amplitude_disturbance * np.sin(2 * np.pi * freq_disturbance * t)
signal_total = noise + disturbance

# Transferencia del micrófono de error
mic_output = butter_lowpass_filter(noise, cutoff=7900, fs=fs, order=6)  # Frecuencia de corte: 20 kHz

# Algoritmo LMS con salida del micrófono
mu = config.get('mu',0.01) #factor de aprendizaje: Uno más bajo aumenta la estabilidad, disminuyendo la performance del ANC
filter_order = int(config.get('fo',32))
reference_signal = noise
output_signal, error_signal, _ = lms_filter(mic_output, reference_signal, mu, filter_order)

# Transferencia del altavoz de cancelación
speaker_output = butter_lowpass_filter(output_signal, cutoff=7900, fs=fs, order=6)  # Frecuencia de corte: 10 kHz

# Señal después del sistema ANC
anc_output = mic_output - speaker_output

# Gráficos
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, mic_output, label="Señal captada (Micrófono de Error)", color="red")
plt.title("Señal de Entrada (Micrófono de Error)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, speaker_output, label="Señal Antifase (Altavoz)", color="blue")
plt.title("Señal Antifase Generada por el Altavoz")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, anc_output, label="Señal Residual (Ruido Cancelado)", color="green")
plt.title("Señal Residual")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
