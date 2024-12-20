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
def butter_lowpass(cutoff, fs, order):
    """
    Crea un filtro pasa bajas Butterworth.
    - cutoff: Frecuencia de corte del filtro(Hz).
    - fs: Frecuencia de muestreo (Hz).
    - order: Orden del filtro. Valores mayores generan transiciones más nítidas.
    - return: Coeficientes del filtro.
    """
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False) 
    return b, a

# Algoritmo LMS (filtro adaptativo)
def lms_filter(desired, x, mu, filter_order):
    """
    Algoritmo LMS para filtrado adaptativo.
    - desired: Señal de referencia (ideal).
    - x: Señal de entrada.
    - mu: Factor de aprendizaje: controla la rapidez del ajuste de los coeficientes.
    - filter_order: Orden del filtro: cantidad de coeficientes del filtro a ajustar.
    - return: Salida del filtro, error y coeficientes del filtro (ignored).
    """

    N = len(x) # Longitud de la señal de entrada
    w = np.zeros(filter_order)  # Coeficientes iniciales del filtro
    y = np.zeros(N)  # Salida del filtro
    e = np.zeros(N)  # Error
    
    for n in range(filter_order, N):
        x_n = x[n-filter_order:n][::-1]  # Ventana de entrada: toma una sección de la señal x invertida (x_n = [x[n-1], x[n-2], ..., x[n-filter_order]])
        y[n] = np.dot(w, x_n)  # Salida del filtro: producto punto o interno entre los coeficientes del filtro(w) y la ventana de entrada(x_n)
        e[n] = desired[n] - y[n]  # Error: diferencia entre la señal deseada y la salida del filtro (salida calculada)
        w += 2 * mu * e[n] * x_n  # Actualización de coeficientes: los coeficientes w del filtro se ajustan en función del error y la ventana de entrada.
    return y, e, w # Salida del filtro, error y coeficientes del filtro

def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Aplica un filtro pasa bajas Butterworth a los datos.
    - data: Señal de entrada a filtrar.
    - cutoff: Frecuencia de corte del filtro(Hz).
    - fs: Frecuencia de muestreo (Hz).
    - order: Orden del filtro. Valores mayores generan transiciones más nítidas.
    - return: Señal filtrada.
    """
    b, a = butter_lowpass(cutoff, fs, order=order) 
    y = lfilter(b, a, data) 
    return y

# Lista de archivos config disponibles: Casos de uso
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
t = np.linspace(0, duration, int(fs * duration), endpoint=False) # Genera un vector de tiempo t para las señales

# Configuración de ruido y perturbación
freq_noise = config.get('freq_noise', 0)
amplitude_noise = config.get('amplitude_noise', 0)
freq_disturbance = config.get('freq_disturbance', 0)
amplitude_disturbance = config.get('amplitude_disturbance', 0)

# Generar ruido y perturbaciones
noise = amplitude_noise * np.sin(2 * np.pi * freq_noise * t) # Señal de referencia basada en la amplitud y frecuencia configuradas.
disturbance = amplitude_disturbance * np.sin(2 * np.pi * freq_disturbance * t) # Señal de perturbación adicional.
signal_total = noise + disturbance # Punto suma: combinación de referencia y perturbación: señal total que llega al micrófono de error.

# Transferencia del micrófono de error: filtra la señal total para simular la respuesta del micrófono.
mic_output = butter_lowpass_filter(signal_total, cutoff=7900, fs=fs, order=6)  # Frecuencia de corte original: 20 kHz

# Algoritmo LMS con salida del micrófono
mu = config.get('mu',0.01) #factor de aprendizaje: Uno más bajo aumenta la estabilidad, disminuyendo la performance del ANC
filter_order = int(config.get('fo',32)) # Orden del filtro: cantidad de coeficientes del filtro a ajustar.
reference_signal = signal_total # Señal de referencia: señal total que llega al micrófono de error.
output_signal, error_signal, _ = lms_filter(mic_output, reference_signal, mu, filter_order) 

# Transferencia del altavoz de cancelación: filtra la señal de salida del filtro LMS para simular la respuesta del altavoz.
speaker_output = butter_lowpass_filter(output_signal, cutoff=7900, fs=fs, order=6)  # Frecuencia de corte original: 10 kHz

# Señal residual después del sistema ANC:  
anc_output = mic_output - speaker_output # Señal residual: diferencia entre la señal captada por el micrófono de error y la señal antifase generada por el altavoz.

# Gráficos
#1: Señal captada por el micrófono de error.
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, mic_output, label="Señal captada (Micrófono de Error)", color="red")
plt.title("Señal de Entrada (Micrófono de Error)")
plt.legend()
plt.grid()

#2: Señal antifase generada por el altavoz.
plt.subplot(3, 1, 2)
plt.plot(t, speaker_output, label="Señal Antifase (Altavoz)", color="blue")
plt.title("Señal Antifase Generada por el Altavoz")
plt.legend()
plt.grid()

#3: Señal residual tras la cancelación activa del sistema ANC. 
plt.subplot(3, 1, 3)
plt.plot(t, anc_output, label="Señal Residual (Ruido Cancelado)", color="green")
plt.title("Señal Residual")
plt.legend()
plt.grid()

# Ajustar diseño de gráficos
plt.tight_layout() #Evita superposiciones
plt.show() 
