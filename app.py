import os
import sys
import time
import datetime
import glob
import json
from typing import Optional, Dict

# Importar para abrir el navegador
import webbrowser

# Configurar torchaudio para usar soundfile en lugar de torchcodec
# Esto evita el error de torchcodec que requiere FFmpeg
try:
    import torch
    import torchaudio
    import numpy as np
    import soundfile as sf
    
    # Parchear torchaudio.load para usar soundfile directamente
    original_torchaudio_load = torchaudio.load
    def load_with_soundfile(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kwargs):
        try:
            # Leer con soundfile
            data, sample_rate = sf.read(filepath, start=frame_offset, frames=num_frames, dtype='float32', always_2d=True)
            # Convertir a tensor forzando Float32 para compatibilidad con cuFFT
            if channels_first and data.ndim > 1:
                data = data.T
            tensor = torch.from_numpy(data.copy()).to(torch.float32)
            return tensor, sample_rate
        except Exception as e:
            # Fallback a la función original si soundfile falla
            return original_torchaudio_load(filepath, frame_offset=frame_offset, num_frames=num_frames, 
                                          normalize=normalize, channels_first=channels_first, **kwargs)
    
    torchaudio.load = load_with_soundfile
except Exception as e:
    print(f"Warning: Could not configure torchaudio backend: {e}")

# --- Dependencias de Coqui TTS ---
from TTS.api import TTS
# Bypass de seguridad de PyTorch (esencial)
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

torch.serialization.add_safe_globals([XttsAudioConfig, XttsConfig, BaseDatasetConfig, XttsArgs])
# ---------------------------------

# Dependencias para reproducción en PC
import winsound
from scipy.io.wavfile import read as wav_read
import numpy as np

# --- Dependencias de Flask ---
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
import logging
from colorama import Fore, Style, init

# Inicializar colorama para colores en la consola
init(autoreset=True)

# Desactivar logs de acceso de Flask para mantener la consola limpia
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- CONFIGURACIÓN CENTRALIZADA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated")

# ⚠️ Variables Globales
CURRENT_SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
LAST_USED_SAMPLE = None  # Último sample usado (o primer sample encontrado)
LAST_OVERWRITE_MODE = False  # ⬅️ CLAVE: Variable Global para la persistencia del modo de guardado

CONFIG: Dict = {
    "TTS_MODEL_NAME": "tts_models/multilingual/multi-dataset/xtts_v2",
    "TARGET_LANGUAGE": "es",
    "OUTPUT_FILENAME": "audio_generado_coqui.wav",

    # Parámetros por defecto para la generación
    "DEFAULT_SPEED": 1.05,
    "DEFAULT_TEMPERATURE": 0.05,
    "DEFAULT_REPETITION_PENALTY": 7.0,
    "DEFAULT_TOP_P": 0.75,
    "DEFAULT_TOP_K": 40,

    "AVAILABLE_LANGUAGES": [
        "en", "es", "fr", "de", "it", "pt", "pl", "zh-cn",
        "ja", "ko", "ru", "tr", "hu", "nl", "da", "fi", "sv"
    ]
}

# Variables globales para sincronización UI-API (debe ir después de CONFIG)
UI_STATE: Dict = {
    "speed": CONFIG["DEFAULT_SPEED"],
    "temperature": CONFIG["DEFAULT_TEMPERATURE"],
    "repetition_penalty": CONFIG["DEFAULT_REPETITION_PENALTY"],
    "top_p": CONFIG["DEFAULT_TOP_P"],
    "top_k": CONFIG["DEFAULT_TOP_K"],
    "language": CONFIG["TARGET_LANGUAGE"],
    "prefix": "audio_generado"
}

# Inicialización del modelo TTS
tts_api = None

# Sistema de logs en tiempo real
import queue
import threading
log_queue = queue.Queue()
log_listeners = []

# 🔒 BLOQUEO DE SEGURIDAD PARA GPU
# Asegura que solo una generación ocurra a la vez, evitando errores de CUDA por concurrencia.
tts_lock = threading.Lock()


# Función para agregar logs al sistema
def add_log(log_type, message, data=None):
    """Agrega un log al sistema en tiempo real."""
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": log_type,  # 'api_request', 'api_response', 'api_error', 'info', 'success', 'error'
        "message": message,
        "data": data or {}
    }
    log_queue.put(log_entry)
    # También imprimir en consola
    color_map = {
        'api_request': Fore.CYAN,
        'api_response': Fore.GREEN,
        'api_error': Fore.RED,
        'info': Fore.BLUE,
        'success': Fore.GREEN,
        'error': Fore.RED,
        'ui_notification': Fore.MAGENTA,
        'generated_by_api': Fore.MAGENTA
    }
    color = color_map.get(log_type, Fore.WHITE)
    print(f"{color}[{log_entry['timestamp']}] {message}{Style.RESET_ALL}")


# --- FUNCIONES DE UTILIDAD ---

def get_precision_mode():
    """Lee la configuración de precisión del archivo externo."""
    precision_file = os.path.join(BASE_DIR, "precision_config.txt")
    if os.path.exists(precision_file):
        try:
            with open(precision_file, "r") as f:
                for line in f:
                    if line.startswith("PRECISION="):
                        return line.split("=")[1].strip().upper()
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ No se pudo leer precision_config.txt: {e}. Usando Float32 por defecto.{Style.RESET_ALL}")
    return "FP16"


def find_speaker_samples(directory: str) -> list:
    """Busca archivos de audio en el directorio especificado (múltiples extensiones)."""
    if not os.path.isdir(directory):
        return []

    samples = []
    # Extensiones de audio comunes
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a', '*.aac', '*.wma']
    for ext in audio_extensions:
        search_path = os.path.join(directory, ext)
        found_files = glob.glob(search_path)
        samples.extend([os.path.basename(f) for f in found_files])

    return sorted(list(set(samples)))


def initialize_tts():
    """Carga el modelo XTTS-v2."""
    global tts_api
    if tts_api is None:
        precision_mode = get_precision_mode()
        print(f"{Fore.YELLOW}🔨 Inicializando modelo Coqui TTS ({CONFIG['TTS_MODEL_NAME']}) en modo {precision_mode}. Esto puede tardar...")
        try:
            tts_api = TTS(model_name=CONFIG["TTS_MODEL_NAME"])
            # Mover modelo a GPU si está disponible (método recomendado)
            if torch.cuda.is_available():
                try:
                    # Aplicar precisión según configuración
                    if precision_mode == "FP16":
                        tts_api = tts_api.to(device='cuda', dtype=torch.float16)
                        print(f"{Fore.GREEN}✅ [MODO FP16] Modelo XTTS-v2 cargado en FP16 (Media Precisión) y movido a GPU.{Style.RESET_ALL}")
                    else:
                        # FLUJO ORIGINAL: FP32
                        tts_api = tts_api.to('cuda')
                        print(f"{Fore.GREEN}✅ [MODO FP32] Modelo XTTS-v2 cargado con éxito y movido a GPU (Float32 original).{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠️  Modelo cargado pero no se pudo mover a GPU o aplicar precisión: {e}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}✅ Modelo XTTS-v2 cargado con éxito (usando CPU).{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}✅ Modelo XTTS-v2 cargado con éxito (usando CPU).{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ ERROR al inicializar Coqui TTS: {e}{Style.RESET_ALL}")
            raise e
    return tts_api


def play_audio_backend(audio_path):
    """
    Reproduce un archivo de audio WAV directamente en el PC donde corre el script.
    Usa winsound (nativo de Windows) para mayor estabilidad en hilos.
    """
    try:
        # Reproducir usando winsound (SND_FILENAME reproduce el archivo por su ruta)
        # SND_NODEFAULT evita que suene un 'ding' si el archivo no se encuentra
        print(f"{Fore.CYAN}🔊 REPRODUCIENDO: {os.path.basename(audio_path)}{Style.RESET_ALL}")
        winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)

    except Exception as e:
        print(f"{Fore.RED}❌ ERROR DE REPRODUCCIÓN (winsound): {e}{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}Asegúrate de que el archivo sea un WAV válido y que tu PC tenga un dispositivo de audio activo.{Style.RESET_ALL}")


# Añadir el parámetro should_overwrite
def generate_tts(text, sample_filename, language, speed, temperature, repetition_penalty, top_p, top_k, prefix,
                 should_overwrite: bool = False):
    """Lógica central de generación de TTS (compartida por UI y API)."""
    tts_api_instance = initialize_tts()

    if tts_api_instance is None:
        raise Exception("Modelo TTS no cargado.")

    speaker_sample_path = os.path.join(CURRENT_SAMPLES_DIR, sample_filename)
    if not os.path.exists(speaker_sample_path):
        raise FileNotFoundError(f"Muestra de voz no encontrada en: {speaker_sample_path}")

    # Lógica de Sobrescritura
    if should_overwrite:
        # Usa el nombre fijo (prefix.wav) para sobrescribir y ahorrar espacio
        unique_filename = f"{prefix}.wav"
        print(f"{Fore.RED}⚠️ [Save Mode] Sobrescritura forzada: {unique_filename}{Style.RESET_ALL}")
    else:
        # Comportamiento anterior: Nombre único con timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{prefix}_{timestamp}.wav"

    audio_path = os.path.join(GENERATED_DIR, unique_filename)

    start_time = time.time()

    # 🔒 SINCRONIZACIÓN: Solo un hilo puede usar la GPU a la vez
    with tts_lock:
        precision_mode = get_precision_mode()
        
        # Selección de contexto: Solo usar FP16 si está configurado y hay GPU
        if precision_mode == "FP16" and torch.cuda.is_available():
            autocast_context = torch.amp.autocast('cuda', dtype=torch.float16)
            use_autocast = True
        else:
            # FLUJO ORIGINAL: Sin autocast forzado (o deshabilitado explícitamente)
            autocast_context = torch.amp.autocast('cuda', enabled=False)
            use_autocast = False

        if use_autocast:
            with autocast_context:
                tts_api_instance.tts_to_file(
                    text=text,
                    file_path=audio_path,
                    speaker_wav=speaker_sample_path,
                    language=language,
                    speed=speed,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    top_k=top_k,
                    enable_text_splitting=True
                )
        else:
            # Flujo idéntico al original
            tts_api_instance.tts_to_file(
                text=text,
                file_path=audio_path,
                speaker_wav=speaker_sample_path,
                language=language,
                speed=speed,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                enable_text_splitting=True
            )
        
        # Limpiar caché de CUDA para evitar fragmentación (opcional pero recomendado)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    synthesis_time = time.time() - start_time

    # Verificar que el archivo se generó correctamente
    if not os.path.exists(audio_path):
        raise Exception(f"El archivo de audio no se generó en: {audio_path}")
    
    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise Exception(f"El archivo de audio generado está vacío (0 bytes): {audio_path}")

    # Obtener duración del audio
    duration = 0
    try:
        data, samplerate = sf.read(audio_path)
        duration = len(data) / samplerate
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ No se pudo obtener la duración: {e}{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}✅ Archivo generado: {unique_filename} ({file_size} bytes, synth: {synthesis_time:.2f}s, duration: {duration:.2f}s){Style.RESET_ALL}")
    return unique_filename, synthesis_time, duration, audio_path


# --- CÓDIGO FLASK ---

app = Flask(__name__)


# --- RUTAS DE LA INTERFAZ (UI) ---

@app.route('/')
def index():
    """Ruta principal: Muestra la interfaz y lista los samples disponibles."""
    global LAST_USED_SAMPLE
    os.makedirs(CURRENT_SAMPLES_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    samples = find_speaker_samples(CURRENT_SAMPLES_DIR)

    # ⭐️ INICIALIZACIÓN: Establecer el primer sample como default si no hay uno guardado.
    if LAST_USED_SAMPLE is None and samples:
        LAST_USED_SAMPLE = samples[0]

    # La lista de archivos generados se ordena por fecha de creación (más reciente primero)
    generated_files = sorted(
        [f for f in os.listdir(GENERATED_DIR) if f.endswith('.wav') or f.endswith('.mp3')],
        key=lambda x: os.path.getmtime(os.path.join(GENERATED_DIR, x)),
        reverse=True
    )
    
    # Determinar el origen de cada archivo (API o UI)
    def get_file_source(filename):
        """Determina si un archivo fue generado por API o UI."""
        if filename.startswith('api_audio_generated'):
            return 'API'
        return 'UI'

    return render_template(
        'index.html',
        samples=samples,
        # Seleccionar el sample usado por última vez para que el dropdown lo muestre
        last_used_sample=LAST_USED_SAMPLE if LAST_USED_SAMPLE in samples else (samples[0] if samples else ''),
        generated_files=generated_files,
        config=CONFIG,
        current_samples_dir=CURRENT_SAMPLES_DIR
    )


# 🟢 RUTA: Actualiza el modo de sobrescritura inmediatamente al mover el switch. (Solo para la UI)
@app.route('/set_overwrite_mode', methods=['POST'])
def set_overwrite_mode_route():
    """Actualiza la variable global LAST_OVERWRITE_MODE inmediatamente al cambiar el switch."""
    global LAST_OVERWRITE_MODE
    data = request.json

    # El valor 'overwrite' se recibe directamente del switch de la UI.
    new_mode = data.get('overwrite', False)

    LAST_OVERWRITE_MODE = new_mode
    print(f"{Fore.BLUE}🔧 [UI State Change] LAST_OVERWRITE_MODE actualizado a: {new_mode}{Style.RESET_ALL}")

    return jsonify({"success": True, "overwrite_mode": new_mode})


# RUTA para actualizar el sample al cambiar el dropdown
@app.route('/set_last_sample', methods=['POST'])
def set_last_sample_route():
    """Actualiza la variable global LAST_USED_SAMPLE inmediatamente al cambiar el dropdown."""
    global LAST_USED_SAMPLE
    data = request.json
    new_sample = data.get('sample')

    if new_sample:
        # ⭐️ Actualiza la variable global
        LAST_USED_SAMPLE = new_sample
        print(f"{Fore.BLUE}🔧 [UI Change] LAST_USED_SAMPLE actualizado a: {new_sample}{Style.RESET_ALL}")
        return jsonify({"success": True, "last_sample": new_sample})

    return jsonify({"success": False, "message": "Sample no proporcionado."}), 400


@app.route('/generate_audio', methods=['POST'])
def generate_audio_route():
    """Ruta API utilizada por el botón de la interfaz web (maneja la reproducción en el cliente)."""
    global LAST_USED_SAMPLE, LAST_OVERWRITE_MODE, UI_STATE
    data = request.json

    try:
        # Extracción de parámetros
        text_to_speak = data.get('text', '')
        sample_filename = data.get('sample', '')
        prefix_filename = data.get('prefix', 'audio_generado')
        language = data.get('language', CONFIG["TARGET_LANGUAGE"])

        speed = float(data.get('speed', CONFIG["DEFAULT_SPEED"]))
        temperature = float(data.get('temperature', CONFIG["DEFAULT_TEMPERATURE"]))
        repetition_penalty = float(data.get('repetition_penalty', CONFIG["DEFAULT_REPETITION_PENALTY"]))
        top_p = float(data.get('top_p', CONFIG["DEFAULT_TOP_P"]))
        top_k = int(data.get('top_k', CONFIG["DEFAULT_TOP_K"]))
        
        # ⭐️ Actualizar UI_STATE con los parámetros usados
        UI_STATE['speed'] = speed
        UI_STATE['temperature'] = temperature
        UI_STATE['repetition_penalty'] = repetition_penalty
        UI_STATE['top_p'] = top_p
        UI_STATE['top_k'] = top_k
        UI_STATE['language'] = language
        UI_STATE['prefix'] = prefix_filename

        # ⭐️ Lógica de Sobrescritura para la UI: Usar el valor del formulario directamente
        # Si no viene en el formulario, usar el último modo guardado
        should_overwrite = data.get('overwrite', LAST_OVERWRITE_MODE)
        
        # Actualizar LAST_OVERWRITE_MODE con el valor recibido
        LAST_OVERWRITE_MODE = should_overwrite

        # Si el usuario hace clic en generar, actualizamos el último sample usado
        LAST_USED_SAMPLE = sample_filename

        if not text_to_speak or not sample_filename:
            return jsonify({"success": False, "message": "Faltan datos de texto o muestra de voz."}), 400
        
        print(f"{Fore.CYAN}🎯 [UI] Generando audio: text='{text_to_speak[:50]}...', sample='{sample_filename}', overwrite={should_overwrite}{Style.RESET_ALL}")

        # Pasar los parámetros a generate_tts
        unique_filename, synthesis_time, duration, audio_full_path = generate_tts(
            text_to_speak, sample_filename, language, speed, temperature, repetition_penalty, top_p, top_k,
            prefix_filename, should_overwrite
        )
        
        # Verificar que el archivo existe antes de responder
        if not os.path.exists(audio_full_path):
            raise Exception(f"El archivo generado no existe: {audio_full_path}")
        
        file_size = os.path.getsize(audio_full_path)
        print(f"{Fore.GREEN}✅ [UI] Audio generado exitosamente: {unique_filename} ({file_size} bytes){Style.RESET_ALL}")

        # Verificar que el archivo está completamente escrito antes de notificar
        import time
        max_wait_time = 5.0
        wait_interval = 0.1
        waited = 0.0
        
        while waited < max_wait_time:
            if os.path.exists(audio_full_path):
                file_size = os.path.getsize(audio_full_path)
                if file_size > 0:
                    break
            time.sleep(wait_interval)
            waited += wait_interval
        
        # Notificar al UI que se generó un nuevo audio desde la UI
        add_log('ui_notification', f'Nuevo audio generado desde UI: {unique_filename}', {
            'filename': unique_filename,
            'source': 'UI',
            'synthesis_time': f"{synthesis_time:.2f}s",
            'url': f"/generated/{unique_filename}",
            'action': 'file_generated',
            'file_ready': True  # Indicar que el archivo está completamente listo
        })
        
        return jsonify({
            "success": True,
            "filename": unique_filename,
            "url": f"/generated/{unique_filename}",  # URL para que el JS del UI lo reproduzca
            "time": f"{synthesis_time:.2f}s",
            "duration": duration,
            "source": "UI",  # Indicar que fue generado por UI
            "file_size": file_size
        })

    except Exception as e:
        error_msg = str(e)
        print(f"{Fore.RED}❌ ERROR [UI Generación]: {error_msg}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        add_log('error', f'Error al generar audio desde UI: {error_msg}', {
            'error': error_msg,
            'error_type': type(e).__name__
        })
        return jsonify({"success": False, "message": f"Error de síntesis: {error_msg}"}), 500


@app.route('/set_samples_dir', methods=['POST'])
def set_samples_dir_route():
    """Ruta para actualizar el directorio de samples desde el UI."""
    global CURRENT_SAMPLES_DIR, LAST_USED_SAMPLE
    new_dir = request.json.get('directory', '')

    if os.path.isdir(new_dir):
        CURRENT_SAMPLES_DIR = new_dir
        samples = find_speaker_samples(CURRENT_SAMPLES_DIR)

        # ⭐️ Actualiza LAST_USED_SAMPLE al cambiar de directorio si hay samples nuevos
        if samples:
            LAST_USED_SAMPLE = samples[0]

        print(f"{Fore.CYAN}📁 Directorio de Samples actualizado a: {new_dir}{Style.RESET_ALL}")
        return jsonify({"success": True, "samples_dir": new_dir, "samples": samples})
    else:
        print(f"{Fore.RED}❌ Directorio no válido: {new_dir}{Style.RESET_ALL}")
        return jsonify({"success": False, "message": "El directorio especificado no existe."}), 400


# --- NUEVA RUTA API (para consumo externo) ---

@app.route('/api/generate', methods=['POST'])
def api_generate_route():
    """
    Endpoint para generar audio mediante una llamada API externa.
    Sincronizado con el estado del UI. Si no se proporcionan parámetros, usa los del UI.
    Por defecto NO sobrescribe los archivos (genera archivos únicos con timestamp).
    Solo sobrescribe si se solicita explícitamente con 'overwrite: true' en la petición.
    Por defecto NO reproduce el audio (útil para tests y scripts automatizados).
    Solo reproduce si se solicita explícitamente con 'play_audio: true' en la petición.
    """
    global UI_STATE, LAST_USED_SAMPLE, LAST_OVERWRITE_MODE

    client_ip = request.remote_addr
    print(f"\n{Fore.MAGENTA}*** API CONEXIÓN ABIERTA ***{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}<- Solicitud API de: {client_ip}{Style.RESET_ALL}")
    
    try:
        data = request.json
        if not data:
            raise ValueError("JSON de solicitud vacío o inválido.")

        text_to_speak = data.get('text')

        # 🟢 FIX: Usar el LAST_USED_SAMPLE como DEFAULT si no se especifica.
        sample_filename = data.get('sample', LAST_USED_SAMPLE)

        # ⭐️ Lógica de Sobrescritura para la API: Por defecto NO sobrescribe
        # 1. El prefijo por defecto es 'api_audio_generated', pero puede ser personalizado
        prefix_filename = data.get('prefix', 'api_audio_generated')
        # 2. Solo sobrescribe si se solicita explícitamente con overwrite: true
        should_overwrite = data.get('overwrite', False)

        if should_overwrite:
            print(
                f"{Fore.YELLOW}🔧 [API] Modo de Sobrescritura ACTIVADO. Archivo de salida: {prefix_filename}.wav{Style.RESET_ALL}")
        else:
            print(
                f"{Fore.CYAN}🔧 [API] Modo de Sobrescritura DESACTIVADO. Se generará archivo único con timestamp.{Style.RESET_ALL}")

        # La única verificación obligatoria es que haya texto y que exista AL MENOS un sample.
        if not text_to_speak or not sample_filename:
            raise ValueError(
                "El campo 'text' es obligatorio. Asegúrate de que un 'sample' se haya elegido previamente en el UI o envíalo por API si es la primera llamada.")

        # ⭐️ SINCRONIZACIÓN CON UI: Usar valores del UI si no se proporcionan en la API
        # Parámetros opcionales (usa valores del UI si no se especifican)
        language = data.get('language', UI_STATE.get('language', CONFIG["TARGET_LANGUAGE"]))

        # Parámetros de ajuste (usa valores del UI si no se especifican)
        speed = float(data.get('speed', UI_STATE.get('speed', CONFIG["DEFAULT_SPEED"])))
        temperature = float(data.get('temperature', UI_STATE.get('temperature', CONFIG["DEFAULT_TEMPERATURE"])))
        repetition_penalty = float(data.get('repetition_penalty', UI_STATE.get('repetition_penalty', CONFIG["DEFAULT_REPETITION_PENALTY"])))
        top_p = float(data.get('top_p', UI_STATE.get('top_p', CONFIG["DEFAULT_TOP_P"])))
        top_k = int(data.get('top_k', UI_STATE.get('top_k', CONFIG["DEFAULT_TOP_K"])))
        
        print(f"{Fore.CYAN}🔗 [API] Usando parámetros del UI: speed={speed}, temp={temperature}, lang={language}{Style.RESET_ALL}")
        
        # 🟢 Obtener parámetro play_audio (por defecto False)
        should_play_audio = data.get('play_audio', False)
        
        # Agregar log de petición API con TODOS los detalles (después de obtener todos los parámetros)
        add_log('api_request', f'Petición API recibida desde {client_ip}', {
            'ip': client_ip,
            'method': 'POST',
            'endpoint': '/api/generate',
            'text': text_to_speak[:100] + ('...' if len(text_to_speak) > 100 else ''),  # Primeros 100 caracteres
            'text_length': len(text_to_speak),
            'sample': sample_filename,
            'language': language,
            'speed': speed,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
            'top_p': top_p,
            'top_k': top_k,
            'prefix': prefix_filename,
            'overwrite': should_overwrite,
            'play_audio': should_play_audio,
            'playback_note': 'Audio se reproducirá' if should_play_audio else 'Audio NO se reproducirá (use play_audio: true para activar)'
        })

        # Pasar el parámetro should_overwrite (False por defecto, True solo si se solicita explícitamente)
        unique_filename, synthesis_time, duration, audio_full_path = generate_tts(
            text_to_speak, sample_filename, language, speed, temperature, repetition_penalty, top_p, top_k,
            prefix_filename, should_overwrite  # False por defecto, True solo si overwrite: true en la petición
        )

        # 🟢 REPRODUCCIÓN CONDICIONAL: Solo reproducir si se solicita explícitamente con play_audio: true
        # Por defecto NO reproduce (útil para tests y scripts automatizados)
        if should_play_audio:
            print(f"{Fore.CYAN}🔊 [API] Reproducción de audio ACTIVADA (play_audio: true){Style.RESET_ALL}")
            play_audio_backend(audio_full_path)
            playback_status = "Reproducido"
        else:
            print(f"{Fore.YELLOW}🔇 [API] Reproducción de audio DESACTIVADA (por defecto). Para reproducir, envía 'play_audio: true' en la petición.{Style.RESET_ALL}")
            playback_status = "No reproducido (use 'play_audio: true' para activar)"

        response_url = f"{request.url_root.rstrip('/')}/generated/{unique_filename}"

        print(f"{Fore.GREEN}<- [API ÉXITO] Audio: {unique_filename} ({synthesis_time:.2f}s) - {playback_status}{Style.RESET_ALL}")
        
        # Agregar log de respuesta exitosa con TODOS los detalles
        add_log('api_response', f'Audio generado exitosamente: {unique_filename} - {playback_status}', {
            'filename': unique_filename,
            'synthesis_time': f"{synthesis_time:.2f}s",
            'text': text_to_speak[:100] + ('...' if len(text_to_speak) > 100 else ''),  # Primeros 100 caracteres
            'text_length': len(text_to_speak),
            'sample': sample_filename,
            'language': language,
            'speed': speed,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
            'top_p': top_p,
            'top_k': top_k,
            'source': 'API',  # Indicar que fue generado por API
            'play_audio': should_play_audio,
            'playback_status': playback_status,
            'download_url': response_url
        })
        
        # Verificar que el archivo existe y está completamente escrito antes de notificar
        # Esperar hasta que el archivo esté completamente escrito (sin delays fijos)
        import time
        max_wait_time = 5.0  # Máximo 5 segundos de espera
        wait_interval = 0.1   # Verificar cada 100ms
        waited = 0.0
        
        while waited < max_wait_time:
            if os.path.exists(audio_full_path):
                file_size = os.path.getsize(audio_full_path)
                if file_size > 0:
                    # Archivo existe y tiene contenido, está listo
                    break
            time.sleep(wait_interval)
            waited += wait_interval
        
        if not os.path.exists(audio_full_path) or os.path.getsize(audio_full_path) == 0:
            print(f"{Fore.YELLOW}⚠️ [API] Advertencia: El archivo no está completamente listo después de {waited:.1f}s{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✅ [API] Archivo verificado y listo: {unique_filename} ({os.path.getsize(audio_full_path)} bytes){Style.RESET_ALL}")
        
        # Notificar al UI que se generó un nuevo audio desde la API
        # La notificación solo se envía cuando el archivo está completamente listo
        notification_data = {
            'filename': unique_filename,
            'source': 'API',
            'synthesis_time': f"{synthesis_time:.2f}s",
            'url': response_url,
            'action': 'file_generated',  # Indicar que es una generación de archivo
            'file_ready': True  # Indicar que el archivo está completamente listo
        }
        print(f"{Fore.MAGENTA}📢 [API] Enviando notificación UI: {unique_filename}{Style.RESET_ALL}")
        add_log('ui_notification', f'Nuevo audio generado desde API: {unique_filename}', notification_data)
        print(f"{Fore.MAGENTA}✅ [API] Notificación enviada a la cola de logs{Style.RESET_ALL}")

        overwrite_msg = "Sobrescritura activada" if should_overwrite else "Archivo único generado (sin sobrescritura)"
        playback_msg = "Audio reproducido" if should_play_audio else "Audio no reproducido (use 'play_audio: true' para activar)"
        return jsonify({
            "success": True,
            "message": f"Audio generado con éxito. {playback_msg}. ({overwrite_msg})",
            "filename": unique_filename,
            "synthesis_time": f"{synthesis_time:.2f}s",
            "duration": duration,
            "download_url": response_url,
            "overwrite": should_overwrite,
            "play_audio": should_play_audio,
            "playback_status": playback_status
        })

    except Exception as e:
        error_msg = f"Error API: {e}"
        print(f"{Fore.RED}<- [API ERROR] {error_msg}{Style.RESET_ALL}")
        
        # Obtener datos de la petición si están disponibles para el log de error
        error_details = {
            'error': str(e),
            'ip': client_ip,
            'error_type': type(e).__name__
        }
        
        # Intentar incluir información de la petición si está disponible
        try:
            if request.json:
                data = request.json
                if data.get('text'):
                    error_details['text'] = data.get('text')[:100] + ('...' if len(data.get('text', '')) > 100 else '')
                if data.get('sample'):
                    error_details['sample'] = data.get('sample')
        except:
            pass  # Si no se puede obtener, continuar sin esos datos
        
        # Agregar log de error con detalles
        add_log('api_error', f'Error en petición API: {error_msg}', error_details)
        
        return jsonify({"success": False, "message": error_msg}), 400

    finally:
        print(f"{Fore.MAGENTA}*** API CONEXIÓN CERRADA ***{Style.RESET_ALL}\n")


# Ruta para servir los archivos generados
@app.route('/generated/<path:filename>')
def generated_files(filename):
    """Permite al navegador acceder a los archivos de audio en la carpeta generated."""
    return send_from_directory(GENERATED_DIR, filename)


# Ruta para eliminar un archivo generado
@app.route('/delete_generated/<path:filename>', methods=['DELETE'])
def delete_generated_file(filename):
    """Elimina un archivo generado."""
    try:
        filepath = os.path.join(GENERATED_DIR, filename)
        
        # Verificar que el archivo existe y está en el directorio correcto (seguridad)
        if not os.path.exists(filepath):
            return jsonify({"success": False, "message": "Archivo no encontrado."}), 404
        
        # Verificar que está dentro del directorio generated (prevenir path traversal)
        if not os.path.abspath(filepath).startswith(os.path.abspath(GENERATED_DIR)):
            return jsonify({"success": False, "message": "Ruta no válida."}), 400
        
        os.remove(filepath)
        print(f"{Fore.YELLOW}🗑️ Archivo eliminado: {filename}{Style.RESET_ALL}")
        add_log('info', f'Archivo eliminado: {filename}')
        
        # Notificar al UI que se eliminó un archivo
        add_log('ui_notification', f'Archivo eliminado: {filename}', {
            'filename': filename,
            'source': 'UI',  # Puede ser UI o API, pero la acción es desde UI
            'action': 'file_deleted'  # Indicar que es una eliminación de archivo
        })
        
        return jsonify({"success": True, "message": f"Archivo '{filename}' eliminado correctamente."})
        
    except Exception as e:
        print(f"{Fore.RED}❌ ERROR al eliminar archivo: {e}{Style.RESET_ALL}")
        return jsonify({"success": False, "message": f"Error al eliminar archivo: {e}"}), 500


# Ruta para renombrar un archivo generado
@app.route('/rename_generated', methods=['POST'])
def rename_generated_file():
    """Renombra un archivo generado."""
    try:
        data = request.json
        old_filename = data.get('old_filename')
        new_filename = data.get('new_filename')
        
        if not old_filename or not new_filename:
            return jsonify({"success": False, "message": "Faltan parámetros."}), 400
        
        # Validar que el nuevo nombre no tenga caracteres peligrosos
        if '..' in new_filename or '/' in new_filename or '\\' in new_filename:
            return jsonify({"success": False, "message": "Nombre de archivo no válido."}), 400
        
        old_filepath = os.path.join(GENERATED_DIR, old_filename)
        new_filepath = os.path.join(GENERATED_DIR, new_filename)
        
        # Verificar que el archivo original existe
        if not os.path.exists(old_filepath):
            return jsonify({"success": False, "message": "Archivo no encontrado."}), 404
        
        # Verificar que no existe un archivo con el nuevo nombre
        if os.path.exists(new_filepath):
            return jsonify({"success": False, "message": "Ya existe un archivo con ese nombre."}), 400
        
        # Renombrar
        os.rename(old_filepath, new_filepath)
        print(f"{Fore.GREEN}✏️ Archivo renombrado: {old_filename} -> {new_filename}{Style.RESET_ALL}")
        add_log('info', f'Archivo renombrado: {old_filename} -> {new_filename}')
        
        # Notificar al UI que se renombró un archivo
        add_log('ui_notification', f'Archivo renombrado: {old_filename} -> {new_filename}', {
            'old_filename': old_filename,
            'new_filename': new_filename,
            'source': 'UI',  # La acción es desde UI
            'action': 'file_renamed'  # Indicar que es un renombrado de archivo
        })
        
        return jsonify({
            "success": True,
            "message": f"Archivo renombrado correctamente.",
            "old_filename": old_filename,
            "new_filename": new_filename
        })
        
    except Exception as e:
        print(f"{Fore.RED}❌ ERROR al renombrar archivo: {e}{Style.RESET_ALL}")
        return jsonify({"success": False, "message": f"Error al renombrar archivo: {e}"}), 500


# Ruta para obtener la lista actualizada de archivos generados
@app.route('/api/generated_files', methods=['GET'])
def get_generated_files_route():
    """Obtiene la lista actualizada de archivos generados con su origen."""
    try:
        generated_files = sorted(
            [f for f in os.listdir(GENERATED_DIR) if f.endswith('.wav') or f.endswith('.mp3')],
            key=lambda x: os.path.getmtime(os.path.join(GENERATED_DIR, x)),
            reverse=True
        )
        
        # Determinar el origen de cada archivo
        files_with_source = []
        for filename in generated_files:
            source = 'API' if filename.startswith('api_audio_generated') else 'UI'
            files_with_source.append({
                'filename': filename,
                'source': source,
                'url': f'/generated/{filename}',
                'modified': os.path.getmtime(os.path.join(GENERATED_DIR, filename))
            })
        
        return jsonify({
            "success": True,
            "files": files_with_source,
            "count": len(files_with_source)
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# Ruta para recibir logs en tiempo real (Server-Sent Events)
@app.route('/api/logs/stream')
def stream_logs():
    """Stream de logs en tiempo real usando Server-Sent Events."""
    def generate():
        while True:
            try:
                # Obtener log del queue (timeout de 1 segundo)
                log_entry = log_queue.get(timeout=1)
                yield f"data: {json.dumps(log_entry)}\n\n"
            except queue.Empty:
                # Enviar un heartbeat para mantener la conexión viva
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# Ruta para servir los archivos de samples (para reproducción en el UI)
@app.route('/samples/<path:filename>')
def sample_files(filename):
    """Permite al navegador acceder a los archivos de audio en la carpeta samples."""
    return send_from_directory(CURRENT_SAMPLES_DIR, filename)


# Ruta para subir nuevos archivos de audio como muestras
@app.route('/upload_sample', methods=['POST'])
def upload_sample_route():
    """Permite subir archivos de audio como nuevas muestras."""
    global LAST_USED_SAMPLE
    
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No se proporcionó ningún archivo."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No se seleccionó ningún archivo."}), 400
    
    # Verificar extensión de audio
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({
            "success": False, 
            "message": f"Extensión no permitida: {file_ext}. Extensiones permitidas: {', '.join(allowed_extensions)}"
        }), 400
    
    try:
        # Asegurar que el directorio existe
        os.makedirs(CURRENT_SAMPLES_DIR, exist_ok=True)
        
        # Guardar el archivo
        filepath = os.path.join(CURRENT_SAMPLES_DIR, file.filename)
        file.save(filepath)
        
        # Actualizar LAST_USED_SAMPLE si es el primer archivo
        if LAST_USED_SAMPLE is None:
            LAST_USED_SAMPLE = file.filename
        
        # Obtener lista actualizada de samples
        samples = find_speaker_samples(CURRENT_SAMPLES_DIR)
        
        print(f"{Fore.GREEN}✅ Archivo subido: {file.filename}{Style.RESET_ALL}")
        
        return jsonify({
            "success": True,
            "message": f"Archivo '{file.filename}' subido correctamente.",
            "filename": file.filename,
            "samples": samples
        })
        
    except Exception as e:
        print(f"{Fore.RED}❌ ERROR al subir archivo: {e}{Style.RESET_ALL}")
        return jsonify({"success": False, "message": f"Error al subir archivo: {e}"}), 500


# Ruta para actualizar parámetros del UI (sincronización con API)
@app.route('/update_ui_state', methods=['POST'])
def update_ui_state_route():
    """Actualiza el estado del UI para sincronización con la API."""
    global UI_STATE
    data = request.json
    
    if 'speed' in data:
        UI_STATE['speed'] = float(data['speed'])
    if 'temperature' in data:
        UI_STATE['temperature'] = float(data['temperature'])
    if 'repetition_penalty' in data:
        UI_STATE['repetition_penalty'] = float(data['repetition_penalty'])
    if 'top_p' in data:
        UI_STATE['top_p'] = float(data['top_p'])
    if 'top_k' in data:
        UI_STATE['top_k'] = int(data['top_k'])
    if 'language' in data:
        UI_STATE['language'] = data['language']
    if 'prefix' in data:
        UI_STATE['prefix'] = data['prefix']
    
    return jsonify({"success": True, "ui_state": UI_STATE})


# Ruta para obtener el estado actual del UI
@app.route('/get_ui_state', methods=['GET'])
def get_ui_state_route():
    """Obtiene el estado actual del UI para la API."""
    global UI_STATE, LAST_USED_SAMPLE, LAST_OVERWRITE_MODE
    return jsonify({
        "success": True,
        "ui_state": UI_STATE,
        "last_used_sample": LAST_USED_SAMPLE,
        "last_overwrite_mode": LAST_OVERWRITE_MODE
    })


# Punto de entrada de la aplicación
if __name__ == '__main__':
    try:
        os.makedirs(CURRENT_SAMPLES_DIR, exist_ok=True)
        initialize_tts()

        # Configurar la URL para el lanzamiento
        PORT = 5000
        URL = f"http://127.0.0.1:{PORT}"

        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}🤖 TTS ENGINE RUNNING - XTTS-v2{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🌐 UI WEB ACCESIBLE: {URL}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}📡 API ENDPOINT: POST {URL}/api/generate{Style.RESET_ALL}")
        print("=" * 80)
        print(f"\n{Fore.GREEN}✅ Servidor iniciado correctamente{Style.RESET_ALL}\n")

        # Abrir el navegador automáticamente
        webbrowser.open_new_tab(URL)

        app.run(host='0.0.0.0', debug=True, use_reloader=False)

    except Exception as e:
        print(f"{Fore.RED}\n🛑 Error al iniciar la aplicación: {e}{Style.RESET_ALL}")