# ASISTENTE AZEMBI - SISTEMA OPTIMIZADO CON ANÁLISIS REAL DE YOUTUBE
# ============================================================================
# Librerías opcionales para funcionalidad completa:
# pip install yt-dlp           # Para descargar videos de YouTube
# pip install moviepy          # Para extraer audio de videos
# pip install opencv-python    # Para análisis visual de frames
# pip install SpeechRecognition # Para transcripción básica
# pip install openai-whisper   # Para transcripción avanzada
# pip install Pillow           # Para manejo de imágenes
# ============================================================================

import os
import sys
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración inicial
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

# Librerías básicas
import re
import json
import base64
import time
import joblib
import tempfile
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from io import BytesIO
import hashlib
import uuid

# Amazon Bedrock
try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

# Librerías de ML
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Librerías para análisis de video y audio
MEDIA_LIBS_AVAILABLE = True
WHISPER_AVAILABLE = False
SPEECH_RECOGNITION_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    yt_dlp = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    mp = None

# Verificar disponibilidad general de librerías de media
MEDIA_LIBS_AVAILABLE = CV2_AVAILABLE and YTDLP_AVAILABLE and MOVIEPY_AVAILABLE

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    print("speech_recognition no disponible")
    sr = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("whisper no disponible")
    whisper = None

# ============================================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Asistente Azembi - Sistema Profesional",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Profesional sin emojis
st.markdown("""
<style>
    /* Variables de color */
    :root {
        --primary-color: #0D47A1;
        --primary-light: #1976D2;
        --primary-lighter: #42A5F5;
        --secondary-color: #E3F2FD;
        --accent-color: #FF6B6B;
        --text-primary: #212529;
        --text-secondary: #6C757D;
        --background: #F8F9FA;
        --card-background: #FFFFFF;
        --border-color: #DEE2E6;
        --success-color: #28A745;
        --warning-color: #FFC107;
        --error-color: #DC3545;
    }

    /* Estilos generales */
    .stApp {
        background-color: var(--background);
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 50%, var(--primary-lighter) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Cards y contenedores */
    .analysis-card {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        border: 1px solid var(--border-color);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    /* Métricas */
    .metric-container {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #90CAF9;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Chat interface */
    .chat-container {
        background: var(--card-background);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        height: 600px;
        display: flex;
        flex-direction: column;
    }

    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
        background: var(--background);
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .chat-input-container {
        display: flex;
        gap: 1rem;
    }

    /* Mensajes del chat */
    .message {
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }

    .message-user {
        flex-direction: row-reverse;
    }

    .message-content {
        max-width: 70%;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .message-user .message-content {
        background: var(--primary-color);
        color: white;
        border-bottom-right-radius: 5px;
    }

    .message-assistant .message-content {
        background: white;
        border: 1px solid var(--border-color);
        border-bottom-left-radius: 5px;
    }

    /* Botones */
    .custom-button {
        background: var(--primary-color);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .custom-button:hover {
        background: var(--primary-light);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Tabs mejoradas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
        border-bottom: 2px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background: transparent;
        border: none;
        color: var(--text-secondary);
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: transparent;
        color: var(--primary-color);
        border-bottom: 3px solid var(--primary-color);
    }

    /* Sidebar */
    .css-1d391kg {
        background: var(--card-background);
        padding: 1.5rem;
    }

    /* Progress bar */
    .progress-container {
        background: var(--border-color);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-bar {
        background: linear-gradient(90deg, var(--primary-color), var(--primary-lighter));
        height: 100%;
        transition: width 0.5s ease;
    }

    /* Video analysis card */
    .video-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }

    .video-thumbnail {
        width: 120px;
        height: 90px;
        border-radius: 8px;
        object-fit: cover;
    }

    .video-info {
        flex: 1;
    }

    .video-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }

    .video-stats {
        display: flex;
        gap: 1rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }

    /* Sesiones de chat */
    .chat-session-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background: var(--secondary-color);
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .chat-session-item:hover {
        background: #BBDEFB;
        transform: translateX(5px);
    }

    .chat-session-item.active {
        background: var(--primary-color);
        color: white;
    }
    
    /* VTR Prediction Box */
    .vtr-prediction-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .vtr-value {
        font-size: 3rem;
        font-weight: 700;
        color: #2E7D32;
        margin: 0.5rem 0;
    }
    
    .vtr-category {
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .vtr-high {
        background: #4CAF50;
        color: white;
    }
    
    .vtr-medium {
        background: #FF9800;
        color: white;
    }
    
    .vtr-low {
        background: #F44336;
        color: white;
    }

    /* Animaciones */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CLASES PRINCIPALES OPTIMIZADAS
# ============================================================================

class YouTubeVideoAnalyzer:
    """Analizador completo de videos de YouTube con transcripción y análisis visual"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.whisper_model = None
        self.recognizer = None
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        
        # Inicializar recognizer solo si está disponible
        if SPEECH_RECOGNITION_AVAILABLE and sr:
            self.recognizer = sr.Recognizer()
        
    def initialize_whisper(self):
        """Inicializa el modelo Whisper para transcripción"""
        if not WHISPER_AVAILABLE or not whisper:
            return False
            
        if self.whisper_model is None:
            try:
                self.whisper_model = whisper.load_model("base")
                return True
            except:
                return False
        return True
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extrae el ID del video de YouTube"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def download_video(self, url: str, max_duration: int = 600) -> Optional[Dict]:
        """Descarga video de YouTube"""
        if not YTDLP_AVAILABLE:
            return {'error': 'yt-dlp no está instalado. Instálalo con: pip install yt-dlp'}
            
        video_id = self.extract_video_id(url)
        if not video_id:
            return None
            
        output_path = self.temp_dir / f"{video_id}.mp4"
        audio_path = self.temp_dir / f"{video_id}.wav"
        
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'force_generic_extractor': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                if info.get('duration', 0) > max_duration:
                    return {'error': f'Video demasiado largo ({info["duration"]}s)'}
                
                # Extraer audio para transcripción solo si moviepy está disponible
                if MOVIEPY_AVAILABLE and mp:
                    try:
                        video_clip = mp.VideoFileClip(str(output_path))
                        audio_clip = video_clip.audio
                        audio_clip.write_audiofile(str(audio_path), logger=None)
                        video_clip.close()
                        audio_clip.close()
                    except Exception as e:
                        print(f"Error extrayendo audio: {e}")
                        audio_path = None
                else:
                    audio_path = None
                
                return {
                    'video_path': str(output_path),
                    'audio_path': str(audio_path) if audio_path else None,
                    'info': {
                        'title': info.get('title', ''),
                        'duration': info.get('duration', 0),
                        'views': info.get('view_count', 0),
                        'likes': info.get('like_count', 0),
                        'upload_date': info.get('upload_date', ''),
                        'description': info.get('description', ''),
                        'channel': info.get('channel', ''),
                        'thumbnail': info.get('thumbnail', '')
                    }
                }
        except Exception as e:
            return {'error': str(e)}
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio usando Whisper o speech_recognition"""
        if not audio_path or audio_path == "None":
            return "No se pudo extraer audio del video. Instala moviepy."
            
        try:
            # Intentar con Whisper primero
            if WHISPER_AVAILABLE and self.initialize_whisper() and self.whisper_model:
                result = self.whisper_model.transcribe(audio_path, language="es")
                return result["text"]
            
            # Fallback a speech_recognition
            if SPEECH_RECOGNITION_AVAILABLE and self.recognizer and sr:
                with sr.AudioFile(audio_path) as source:
                    audio = self.recognizer.record(source)
                    try:
                        return self.recognizer.recognize_google(audio, language="es-ES")
                    except:
                        return "No se pudo transcribir el audio con Google Speech"
            
            return "No hay librerías de transcripción disponibles. Instala whisper o speech_recognition."
                    
        except Exception as e:
            return f"Error en transcripción: {str(e)}"
    
    def analyze_video_frames(self, video_path: str, num_frames: int = 10) -> Dict:
        """Analiza frames del video para extraer características visuales y guardar frames clave"""
        if not MEDIA_LIBS_AVAILABLE or not cv2:
            return {'error': 'OpenCV no está disponible'}
            
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return {}
            
            frame_interval = max(1, total_frames // num_frames)
            
            brightness_values = []
            color_dominants = []
            motion_scores = []
            saved_frames = []
            frame_quality_scores = []
            prev_frame = None
            
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                    
                # Brillo promedio
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Color dominante
                pixels = frame.reshape(-1, 3)
                kmeans = cv2.kmeans(np.float32(pixels), 3, None,
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                   10, cv2.KMEANS_RANDOM_CENTERS)
                dominant_color = kmeans[2][0]
                color_dominants.append(dominant_color)
                
                # Detección de movimiento
                motion_score = 0
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_score = np.mean(diff)
                    motion_scores.append(motion_score)
                
                # Calcular calidad del frame (basado en nitidez, contraste, etc.)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                contrast = gray.std()
                
                # Score de calidad combinado
                quality_score = sharpness * 0.6 + contrast * 0.4 + brightness * 0.2
                frame_quality_scores.append((i, quality_score, frame))
                
                # Guardar frame como imagen
                frame_filename = f"frame_{i:06d}.jpg"
                frame_path = self.frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                saved_frames.append({
                    'frame_number': i,
                    'timestamp': i / cap.get(cv2.CAP_PROP_FPS),
                    'path': str(frame_path),
                    'brightness': brightness,
                    'motion_score': motion_score,
                    'quality_score': quality_score
                })
                
                prev_frame = gray
            
            cap.release()
            
            # Seleccionar el mejor frame (mayor calidad)
            best_frame_data = None
            if frame_quality_scores:
                frame_quality_scores.sort(key=lambda x: x[1], reverse=True)
                best_frame_idx, best_score, best_frame = frame_quality_scores[0]
                
                # Guardar el mejor frame
                best_frame_path = self.frames_dir / "best_frame.jpg"
                cv2.imwrite(str(best_frame_path), best_frame)
                
                # Obtener FPS para calcular timestamp
                fps = cap.get(cv2.CAP_PROP_FPS) if 'cap' in locals() else 30.0
                
                best_frame_data = {
                    'frame_number': best_frame_idx,
                    'timestamp': best_frame_idx / fps if fps > 0 else 0,
                    'path': str(best_frame_path),
                    'quality_score': best_score
                }
            
            return {
                'avg_brightness': np.mean(brightness_values) if brightness_values else 0,
                'brightness_std': np.std(brightness_values) if brightness_values else 0,
                'dominant_colors': color_dominants[:3] if color_dominants else [],
                'motion_intensity': np.mean(motion_scores) if motion_scores else 0,
                'scene_changes': len([i for i in range(1, len(motion_scores)) 
                                    if motion_scores[i] > motion_scores[i-1] * 2]) if motion_scores else 0,
                'saved_frames': saved_frames,
                'best_frame': best_frame_data,
                'total_frames_analyzed': len(saved_frames)
            }
        except Exception as e:
            return {'error': f'Error analizando video: {str(e)}'}
    
    def cleanup_temp_files(self):
        """Limpia todos los archivos temporales"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error limpiando directorio temporal: {e}")
    
    def __del__(self):
        """Destructor para limpiar archivos temporales"""
        self.cleanup_temp_files()
    
    def analyze_complete_video(self, url: str) -> Dict:
        """Análisis completo del video: descarga, transcripción y análisis visual"""
        result = {'url': url, 'status': 'processing'}
        
        # Verificar si yt-dlp está disponible
        if not YTDLP_AVAILABLE or not yt_dlp:
            result['status'] = 'error'
            result['error'] = 'yt-dlp no está instalado. Instálalo con: pip install yt-dlp'
            return result
        
        # Si no hay todas las librerías de media, intentar obtener info básica
        if not MEDIA_LIBS_AVAILABLE:
            try:
                # Usar yt-dlp solo para obtener metadata sin descargar
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    result.update({
                        'title': info.get('title', 'Sin título'),
                        'duration': info.get('duration', 0),
                        'views': info.get('view_count', 0),
                        'likes': info.get('like_count', 0),
                        'description': info.get('description', ''),
                        'channel': info.get('channel', ''),
                        'thumbnail': info.get('thumbnail', ''),
                        'status': 'partial',
                        'note': 'Análisis limitado: instala cv2, moviepy para análisis completo'
                    })
                    return result
            except Exception as e:
                result['status'] = 'error'
                result['error'] = f'Error obteniendo información del video: {str(e)}'
                return result
        
        # Descargar video
        download_result = self.download_video(url)
        if not download_result or 'error' in download_result:
            result['status'] = 'error'
            result['error'] = download_result.get('error', 'Error descargando video')
            return result
        
        result.update(download_result['info'])
        
        # Transcribir audio
        if download_result.get('audio_path') and (WHISPER_AVAILABLE or SPEECH_RECOGNITION_AVAILABLE):
            transcription = self.transcribe_audio(download_result['audio_path'])
            result['transcription'] = transcription
            
            # Análisis de texto
            if transcription and len(transcription) > 10 and not transcription.startswith('Error') and not transcription.startswith('No'):
                words = transcription.split()
                result['word_count'] = len(words)
                result['unique_words'] = len(set(words))
                result['lexical_diversity'] = result['unique_words'] / result['word_count'] if result['word_count'] > 0 else 0
        else:
            if not download_result.get('audio_path'):
                result['transcription'] = 'Audio no extraído - instala moviepy'
            else:
                result['transcription'] = 'Transcripción no disponible - instala whisper o speech_recognition'
        
        # Análisis visual
        if 'video_path' in download_result:
            visual_analysis = self.analyze_video_frames(download_result['video_path'])
            if 'error' not in visual_analysis:
                result['visual_analysis'] = visual_analysis
        
        # Limpiar archivos temporales
        try:
            if 'video_path' in download_result and download_result['video_path']:
                Path(download_result['video_path']).unlink()
            if 'audio_path' in download_result and download_result['audio_path']:
                Path(download_result['audio_path']).unlink()
            
            # Limpiar frames guardados
            if 'visual_analysis' in result and 'saved_frames' in result['visual_analysis']:
                for frame_info in result['visual_analysis']['saved_frames']:
                    try:
                        Path(frame_info['path']).unlink()
                    except:
                        pass
            
            # Limpiar el mejor frame
            if 'visual_analysis' in result and 'best_frame' in result['visual_analysis']:
                try:
                    Path(result['visual_analysis']['best_frame']['path']).unlink()
                except:
                    pass
                    
        except Exception as e:
            print(f"Error limpiando archivos temporales: {e}")
        
        result['status'] = 'completed'
        return result

class ChatSessionManager:
    """Gestor de sesiones de chat múltiples"""
    
    def __init__(self):
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
        if 'current_session_id' not in st.session_state:
            self.create_new_session("Chat Principal")
    
    def create_new_session(self, name: str) -> str:
        """Crea una nueva sesión de chat"""
        session_id = str(uuid.uuid4())
        st.session_state.chat_sessions[session_id] = {
            'id': session_id,
            'name': name,
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }
        st.session_state.current_session_id = session_id
        return session_id
    
    def get_current_session(self) -> Dict:
        """Obtiene la sesión actual"""
        return st.session_state.chat_sessions.get(st.session_state.current_session_id, {})
    
    def update_session_activity(self):
        """Actualiza la última actividad de la sesión"""
        if st.session_state.current_session_id in st.session_state.chat_sessions:
            st.session_state.chat_sessions[st.session_state.current_session_id]['last_activity'] = datetime.now().isoformat()
    
    def add_message(self, role: str, content: str):
        """Añade un mensaje a la sesión actual"""
        session = self.get_current_session()
        if session:
            session['messages'].append({
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            })
            self.update_session_activity()
    
    def delete_session(self, session_id: str):
        """Elimina una sesión"""
        if session_id in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[session_id]
            if st.session_state.current_session_id == session_id:
                if st.session_state.chat_sessions:
                    st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]
                else:
                    self.create_new_session("Chat Principal")

class AzembiLLMSystem:
    """Sistema LLM mejorado con análisis de video"""
    
    def __init__(self, df_enhanced=None, aws_region='us-east-1'):
        self.df_enhanced = df_enhanced
        self.model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        self.bedrock_client = None
        self.video_analyzer = YouTubeVideoAnalyzer()
        
        if BEDROCK_AVAILABLE:
            try:
                self.bedrock_client = boto3.client(
                    service_name='bedrock-runtime', 
                    region_name=aws_region,
                    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
                )
            except Exception as e:
                st.error(f"Error inicializando Bedrock: {e}")
    
    def estimate_vtr_from_video(self, video_data: Dict) -> Dict:
        """Estima el VTR potencial basándose en las características del video"""
        if not video_data or video_data.get('status') != 'completed':
            return {'error': 'Datos de video incompletos'}
        
        # Factores base de la campaña
        avg_vtr = 0.2907  # VTR promedio de la campaña
        
        # Inicializar score
        vtr_score = avg_vtr
        factors = []
        
        # Factor 1: Duración (30-60 segundos es óptimo)
        duration = video_data.get('duration', 0)
        if 30 <= duration <= 60:
            vtr_score *= 1.15
            factors.append("Duración óptima (30-60 seg)")
        elif duration < 30:
            vtr_score *= 0.95
            factors.append("Video muy corto")
        else:
            vtr_score *= 0.90
            factors.append("Video muy largo")
        
        # Factor 2: Engagement (likes/views ratio)
        views = video_data.get('views', 1)
        likes = video_data.get('likes', 0)
        engagement_rate = likes / views if views > 0 else 0
        
        if engagement_rate > 0.05:  # > 5% es excelente
            vtr_score *= 1.20
            factors.append("Alto engagement")
        elif engagement_rate > 0.02:  # > 2% es bueno
            vtr_score *= 1.10
            factors.append("Buen engagement")
        
        # Factor 3: Análisis visual
        if 'visual_analysis' in video_data:
            va = video_data['visual_analysis']
            
            # Movimiento (videos dinámicos funcionan mejor)
            motion = va.get('motion_intensity', 0)
            if motion > 10:
                vtr_score *= 1.10
                factors.append("Alto dinamismo visual")
            
            # Cambios de escena
            scene_changes = va.get('scene_changes', 0)
            if scene_changes > 5:
                vtr_score *= 1.05
                factors.append("Múltiples escenas")
        
        # Factor 4: Contenido verbal (si hay transcripción)
        if 'word_count' in video_data and video_data['word_count'] > 50:
            vtr_score *= 1.08
            factors.append("Contenido verbal rico")
        
        # Limitar VTR entre rangos realistas
        vtr_score = max(0.15, min(0.40, vtr_score))
        
        # Categorizar
        if vtr_score >= 0.35:
            categoria = "ALTO POTENCIAL"
        elif vtr_score >= 0.28:
            categoria = "POTENCIAL MEDIO"
        else:
            categoria = "POTENCIAL BAJO"
        
        return {
            'vtr_estimado': vtr_score,
            'categoria': categoria,
            'vs_promedio': ((vtr_score / avg_vtr - 1) * 100),
            'factores_positivos': factors,
            'confianza': 'Media-Alta' if len(factors) >= 3 else 'Media'
        }
    
    def prepare_comprehensive_context(self, include_video_analysis: bool = False) -> str:
        """Prepara contexto completo incluyendo análisis de video si está disponible"""
        if self.df_enhanced is None:
            return "No hay datos disponibles para el análisis."
        
        try:
            df = self.df_enhanced
            context = "=== CONTEXTO DE DATOS DE LA CAMPAÑA AZEMBI ===\n\n"
            
            # Resumen general
            context += f"**Resumen General:**\n"
            context += f"- Registros Totales: {len(df)}\n"
            context += f"- Videos Únicos: {df['Enlace video'].nunique()}\n"
            context += f"- Inversión Total: ${df['Cost'].sum():,.0f} COP\n"
            context += f"- Impresiones Totales: {df['Impressions'].sum():,.0f}\n\n"
            
            # Métricas de rendimiento
            context += f"**Métricas de Rendimiento (VTR):**\n"
            context += f"- VTR Promedio: {df['VTR'].mean():.4f}\n"
            context += f"- VTR Mediana: {df['VTR'].median():.4f}\n"
            context += f"- Rango VTR: [{df['VTR'].min():.4f} - {df['VTR'].max():.4f}]\n\n"
            
            # Top territorios
            if 'Territorio' in df.columns:
                territory_stats = df.groupby('Territorio')['VTR'].mean().sort_values(ascending=False)
                context += "**Top Territorios por VTR:**\n"
                for territorio, vtr in territory_stats.head(5).items():
                    context += f"- {territorio}: {vtr:.4f}\n"
                context += "\n"
            
            # Top temáticas
            if 'Temática' in df.columns:
                theme_stats = df.groupby('Temática')['VTR'].mean().sort_values(ascending=False)
                context += "**Top Temáticas por VTR:**\n"
                for tema, vtr in theme_stats.head(5).items():
                    context += f"- {tema}: {vtr:.4f}\n"
                context += "\n"
            
            # Análisis de video si está disponible
            if include_video_analysis and 'last_video_analysis' in st.session_state:
                video_data = st.session_state.last_video_analysis
                context += "\n**=== ANÁLISIS DETALLADO DEL VIDEO DE YOUTUBE ===**\n"
                
                # Información básica
                context += f"\n**Información Básica:**\n"
                context += f"- Título: {video_data.get('title', 'N/A')}\n"
                context += f"- Canal: {video_data.get('channel', 'N/A')}\n"
                context += f"- Duración: {video_data.get('duration', 0)} segundos\n"
                context += f"- Vistas: {video_data.get('views', 0):,}\n"
                context += f"- Likes: {video_data.get('likes', 0):,}\n"
                context += f"- Fecha de subida: {video_data.get('upload_date', 'N/A')}\n"
                
                # Descripción
                if 'description' in video_data:
                    context += f"\n**Descripción del Video:**\n{video_data['description'][:500]}...\n"
                
                # Transcripción
                if 'transcription' in video_data and not video_data['transcription'].startswith('Error'):
                    context += f"\n**Transcripción del Audio:**\n{video_data['transcription'][:1500]}...\n"
                    if 'word_count' in video_data:
                        context += f"\n- Palabras totales: {video_data['word_count']}\n"
                        context += f"- Palabras únicas: {video_data.get('unique_words', 0)}\n"
                        context += f"- Diversidad léxica: {video_data.get('lexical_diversity', 0):.2f}\n"
                
                # Análisis visual
                if 'visual_analysis' in video_data and 'error' not in video_data['visual_analysis']:
                    va = video_data['visual_analysis']
                    context += f"\n**Análisis Visual Detallado:**\n"
                    context += f"- Brillo promedio: {va.get('avg_brightness', 0):.1f} (escala 0-255)\n"
                    context += f"- Variación de brillo: {va.get('brightness_std', 0):.1f}\n"
                    context += f"- Intensidad de movimiento: {va.get('motion_intensity', 0):.1f}\n"
                    context += f"- Cambios de escena detectados: {va.get('scene_changes', 0)}\n"
                    context += f"- Frames analizados: {va.get('total_frames_analyzed', 0)}\n"
                    
                    if 'best_frame' in va and va['best_frame']:
                        context += f"\n**Frame más representativo:**\n"
                        context += f"- Timestamp: {va['best_frame']['timestamp']:.1f} segundos\n"
                        context += f"- Calidad visual: {va['best_frame']['quality_score']:.2f}\n"
                
                # Estado del análisis
                context += f"\n**Estado del Análisis:** {video_data.get('status', 'desconocido')}\n"
                if 'note' in video_data:
                    context += f"**Nota:** {video_data['note']}\n"
                
                # Estimación de VTR si está disponible
                if 'last_vtr_estimation' in st.session_state:
                    vtr_est = st.session_state.last_vtr_estimation
                    context += f"\n**=== PREDICCIÓN DE VTR ===**\n"
                    context += f"- VTR Estimado: {vtr_est['vtr_estimado']:.4f}\n"
                    context += f"- Categoría: {vtr_est['categoria']}\n"
                    context += f"- Vs. Promedio Campaña: {vtr_est['vs_promedio']:+.1f}%\n"
                    context += f"- Nivel de Confianza: {vtr_est['confianza']}\n"
                    if vtr_est['factores_positivos']:
                        context += f"- Factores Positivos: {', '.join(vtr_est['factores_positivos'])}\n"
            
            context += "\n=== FIN DEL CONTEXTO ===\n"
            return context
        except Exception as e:
            return f"Error preparando contexto: {str(e)}"
    
    def generate_response(self, user_query: str, chat_history: List[Dict], 
                          include_video_context: bool = False) -> str:
        """Genera respuesta usando Claude con contexto mejorado"""
        if not self.bedrock_client:
            return "El sistema de IA no está disponible. Verifica la configuración de AWS."

        # Detectar si hay una URL de YouTube en la consulta
        youtube_url_pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]+)'
        youtube_match = re.search(youtube_url_pattern, user_query)

        video_analysis_result = None
        if youtube_match:
            url = youtube_match.group(1)
            with st.spinner("Analizando video de YouTube... Esto puede tomar unos minutos"):
                # FIX: Call the method from the self.video_analyzer object
                video_analysis = self.video_analyzer.analyze_complete_video(url)
                
                st.session_state.last_video_analysis = video_analysis
                include_video_context = True
                video_analysis_result = video_analysis
                
                # Mostrar el mejor frame si está disponible
                if video_analysis.get('status') == 'completed' and 'visual_analysis' in video_analysis:
                    visual_data = video_analysis['visual_analysis']
                    if 'best_frame' in visual_data and visual_data['best_frame']:
                        best_frame = visual_data['best_frame']
                        try:
                            if PIL_AVAILABLE:
                                from PIL import Image
                                img = Image.open(best_frame['path'])
                                st.image(img, caption=f"Frame más representativo (segundo {best_frame['timestamp']:.1f})", 
                                        use_column_width=True)
                        except Exception as e:
                            st.info(f"Frame representativo en segundo {best_frame['timestamp']:.1f}")
                    
                # Estimar VTR del video
                if video_analysis.get('status') == 'completed':
                    vtr_estimation = self.estimate_vtr_from_video(video_analysis)
                    if 'error' not in vtr_estimation:
                        st.session_state.last_vtr_estimation = vtr_estimation
                        
                        # Mostrar estimación de VTR
                        st.success(f"**VTR Estimado: {vtr_estimation['vtr_estimado']:.4f}** ({vtr_estimation['categoria']})")
                        st.info(f"Comparación con promedio: {vtr_estimation['vs_promedio']:+.1f}%")
                        
                        if vtr_estimation['factores_positivos']:
                            st.write("**Factores positivos detectados:**")
                            for factor in vtr_estimation['factores_positivos']:
                                st.write(f"- {factor}")
        
        # Preparar contexto
        data_context = self.prepare_comprehensive_context(include_video_context)
        
        # System prompt mejorado
        system_prompt = f"""
        Eres el Asistente Azembi, un experto en análisis de campañas de marketing político y predicción de rendimiento de videos.
        
        INSTRUCCIONES CRÍTICAS:
        1. Responde SIEMPRE en español de manera profesional y estructurada
        2. Cuando analices un video de YouTube, SIEMPRE debes proporcionar:
           
           a) **PREDICCIÓN DE VTR** (lo más importante):
              - VTR estimado con 4 decimales (ej: 0.3245)
              - Categoría: ALTO POTENCIAL (>0.35), POTENCIAL MEDIO (0.28-0.35), POTENCIAL BAJO (<0.28)
              - Comparación con el promedio de la campaña (0.2907)
              - Nivel de confianza de la predicción
           
           b) **ANÁLISIS DE FACTORES CLAVE**:
              - Duración del video (óptimo: 30-60 segundos)
              - Engagement (likes/views ratio)
              - Dinamismo visual (movimiento, cambios de escena)
              - Calidad del contenido verbal
              - Temática y relevancia política
           
           c) **RECOMENDACIONES ESPECÍFICAS**:
              - Qué mantener del video actual
              - Qué mejorar para aumentar el VTR
              - Comparación con videos exitosos de la campaña
        
        3. Basa SIEMPRE tus predicciones en los datos reales de la campaña:
           - VTR promedio: 0.2907
           - Mejores temáticas: Seguridad (0.3489), Petro/Corrupción (0.3476)
           - Peores temáticas: Salud (0.1438)
           - Mejores territorios: Antioquia (0.3358), Bogotá (0.3299)
        
        4. El candidato es Miguel Uribe Turbay - considera su imagen y mensaje político
        
        5. Sé específico y cuantitativo en tus análisis - no uses generalidades
        
        FORMATO DE RESPUESTA PARA VIDEOS:
        Inicia siempre con un cuadro resumen:
        ```
        PREDICCIÓN DE VTR
        ================
        VTR Estimado: [número]
        Categoría: [categoría]
        Vs. Promedio: [porcentaje]%
        Confianza: [nivel]
        ```
        
        Luego proporciona el análisis detallado.
        
        CONTEXTO DE DATOS:
        {data_context}
        """
        
        # Construir mensajes
        messages = []
        for msg in chat_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        messages.append({"role": "user", "content": user_query})
        
        # Configurar y llamar al modelo
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
        })
        
        try:
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            ai_response = response_body['content'][0]['text'].strip()
            
            # Si se analizó un video, añadir información visual al final
            if video_analysis_result and video_analysis_result.get('status') == 'completed':
                if 'transcription' in video_analysis_result and not video_analysis_result['transcription'].startswith('Error'):
                    words = video_analysis_result.get('word_count', 0)
                    if words > 50:
                        ai_response += f"\n\n**Nota sobre el contenido:** El video contiene {words} palabras en su narración, lo que indica un contenido rico en información."
            
            return ai_response
            
        except Exception as e:
            return f"Error al generar respuesta: {str(e)}"

class EnhancedMLPipeline:
    """Pipeline de ML optimizado"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results_df = None
    
    def prepare_ml_features(self, df_enhanced):
        """Prepara características para ML"""
        base_features = ['Cost', 'Territorio', 'Sexo', 'Edad', 'Temática']
        video_features = ['duration', 'view_count', 'like_count']
        all_features = base_features + video_features
        return [f for f in all_features if f in df_enhanced.columns]
    
    def train_models(self, df_enhanced, features):
        """Entrena múltiples modelos y selecciona el mejor"""
        X = df_enhanced[features].copy()
        y = df_enhanced['VTR'].copy()
        
        # Limpiar datos
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X, y = X[mask], y[mask]
        
        if len(X) < 10:
            return None, None, None
        
        # Preparar características
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Modelos a entrenar
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        results = {}
        best_score = -np.inf
        
        for name, model in models.items():
            try:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                results[name] = {
                    'R2': r2,
                    'MAE': mae,
                    'RMSE': rmse
                }
                
                if r2 > best_score:
                    best_score = r2
                    self.best_model = pipeline
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error entrenando {name}: {e}")
        
        self.results_df = pd.DataFrame(results).T
        return self.best_model, self.results_df, self.best_model_name

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def load_data_automatically():
    """Carga datos automáticamente"""
    if 'data_loaded' in st.session_state:
        return True
    
    DATA_FILE = 'Datos Proyecto Azembi Deeploy (2).xlsx'
    
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_excel(DATA_FILE, sheet_name='Final')
            
            # Limpiar columnas
            df.columns = df.columns.str.strip()
            
            # Convertir columnas numéricas
            numeric_cols = ['Cost', 'Impressions', 'VTR']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filtrar datos válidos
            df_clean = df.dropna(subset=['VTR', 'Enlace video']).copy()
            
            st.session_state.df = df_clean
            st.session_state.data_loaded = True
            return True
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return False
    
    return False

def create_dashboard_visualizations(df):
    """Crea visualizaciones del dashboard"""
    col1, col2 = st.columns(2)
    
    with col1:
        # VTR por Territorio
        territorio_data = df.groupby('Territorio')['VTR'].agg(['mean', 'count']).reset_index()
        territorio_data = territorio_data.sort_values('mean', ascending=False).head(10)
        
        fig = px.bar(
            territorio_data, 
            x='Territorio', 
            y='mean',
            title='VTR Promedio por Territorio (Top 10)',
            labels={'mean': 'VTR Promedio'},
            color='mean',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # VTR por Temática
        tematica_data = df.groupby('Temática')['VTR'].agg(['mean', 'count']).reset_index()
        tematica_data = tematica_data.sort_values('mean', ascending=False).head(10)
        
        fig = px.bar(
            tematica_data,
            x='Temática',
            y='mean',
            title='VTR Promedio por Temática (Top 10)',
            labels={'mean': 'VTR Promedio'},
            color='mean',
            color_continuous_scale='Greens'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribución de VTR
    fig = px.histogram(
        df,
        x='VTR',
        nbins=50,
        title='Distribución de VTR',
        labels={'count': 'Frecuencia', 'VTR': 'View-Through Rate'},
        color_discrete_sequence=['#1976D2']
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>Asistente Azembi</h1>
    <p>Sistema Profesional de Análisis de Campañas con IA</p>
</div>
""", unsafe_allow_html=True)

# Inicializar gestores
chat_manager = ChatSessionManager()

# Sidebar
with st.sidebar:
    st.markdown("### Panel de Control")
    
    # Estado del sistema
    with st.expander("Estado del Sistema", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bedrock", "Activo" if BEDROCK_AVAILABLE else "Inactivo")
            st.metric("OpenCV", "Activo" if MEDIA_LIBS_AVAILABLE else "Inactivo")
        with col2:
            st.metric("Whisper", "Activo" if WHISPER_AVAILABLE else "Inactivo")
            st.metric("Speech Rec", "Activo" if SPEECH_RECOGNITION_AVAILABLE else "Inactivo")
    
    # Gestión de sesiones de chat
    st.markdown("### Sesiones de Chat")
    
    # Nueva sesión
    new_session_name = st.text_input("Nombre de nueva sesión")
    if st.button("Crear Nueva Sesión", use_container_width=True):
        if new_session_name:
            chat_manager.create_new_session(new_session_name)
            st.rerun()
    
    # Lista de sesiones
    st.markdown("#### Sesiones Activas")
    for session_id, session in st.session_state.chat_sessions.items():
        is_current = session_id == st.session_state.current_session_id
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                session['name'],
                key=f"session_{session_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                st.session_state.current_session_id = session_id
                st.rerun()
        
        with col2:
            if st.button("🗑", key=f"delete_{session_id}"):
                chat_manager.delete_session(session_id)
                st.rerun()
    
    st.divider()
    
    # Cargar datos
    if st.button("Cargar Datos", type="primary", use_container_width=True):
        if load_data_automatically():
            st.success("Datos cargados correctamente")
            
            # Entrenar modelos
            ml_pipeline = EnhancedMLPipeline()
            features = ml_pipeline.prepare_ml_features(st.session_state.df)
            
            with st.spinner("Entrenando modelos..."):
                model, results, best_name = ml_pipeline.train_models(
                    st.session_state.df, features
                )
                
                if model:
                    st.session_state.ml_pipeline = ml_pipeline
                    st.session_state.ml_trained = True
                    st.success(f"Mejor modelo: {best_name}")
                    
                    # Inicializar LLM
                    st.session_state.llm_system = AzembiLLMSystem(st.session_state.df)
                    st.session_state.llm_ready = True

# Contenido principal
if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    tabs = st.tabs([
        "Chat Inteligente",
        "Dashboard",
        "Predicción VTR",
        "Análisis ML"
    ])
    
    # Tab 1: Chat
    with tabs[0]:
        current_session = chat_manager.get_current_session()
        
        st.markdown(f"### {current_session.get('name', 'Chat')}")
        
        # Contenedor de mensajes
        chat_container = st.container()
        
        with chat_container:
            for message in current_session.get('messages', []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Input de chat
        if prompt := st.chat_input("Escribe tu mensaje o pega un link de YouTube para analizar..."):
            # Añadir mensaje del usuario
            chat_manager.add_message("user", prompt)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generar respuesta
            with st.chat_message("assistant"):
                if 'llm_system' in st.session_state:
                    # Detectar si es un video de YouTube
                    youtube_url_pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]+)'
                    is_youtube = re.search(youtube_url_pattern, prompt)
                    
                    if is_youtube:
                        st.info("Analizando video de YouTube... Esto puede tomar 1-2 minutos")
                    
                    response = st.session_state.llm_system.generate_response(
                        prompt,
                        current_session.get('messages', [])
                    )
                    
                    # Si se analizó un video, mostrar información adicional
                    if is_youtube and 'last_video_analysis' in st.session_state:
                        video_data = st.session_state.last_video_analysis
                        
                        # Crear expander con detalles del video
                        with st.expander("Ver detalles completos del análisis", expanded=False):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Métricas del Video:**")
                                st.write(f"- Duración: {video_data.get('duration', 0)} segundos")
                                st.write(f"- Vistas: {video_data.get('views', 0):,}")
                                st.write(f"- Likes: {video_data.get('likes', 0):,}")
                                st.write(f"- Canal: {video_data.get('channel', 'N/A')}")
                                
                                if 'word_count' in video_data:
                                    st.markdown("**Análisis de Texto:**")
                                    st.write(f"- Palabras: {video_data['word_count']}")
                                    st.write(f"- Diversidad léxica: {video_data.get('lexical_diversity', 0):.2%}")
                            
                            with col2:
                                if 'visual_analysis' in video_data and 'error' not in video_data['visual_analysis']:
                                    va = video_data['visual_analysis']
                                    st.markdown("**Análisis Visual:**")
                                    st.write(f"- Brillo promedio: {va.get('avg_brightness', 0):.1f}")
                                    st.write(f"- Movimiento: {va.get('motion_intensity', 0):.1f}")
                                    st.write(f"- Cambios de escena: {va.get('scene_changes', 0)}")
                                    
                                    # Mostrar frames guardados si existen
                                    if 'saved_frames' in va and va['saved_frames']:
                                        st.markdown("**Frames clave del video:**")
                                        frames_to_show = va['saved_frames'][:3]  # Mostrar máximo 3
                                        cols = st.columns(len(frames_to_show))
                                        for idx, (col, frame_info) in enumerate(zip(cols, frames_to_show)):
                                            with col:
                                                try:
                                                    if PIL_AVAILABLE:
                                                        from PIL import Image
                                                        img = Image.open(frame_info['path'])
                                                        st.image(img, caption=f"Seg {frame_info['timestamp']:.1f}")
                                                except:
                                                    st.write(f"Frame en seg {frame_info['timestamp']:.1f}")
                else:
                    response = "El sistema de IA no está inicializado. Por favor, carga los datos primero."
                
                st.markdown(response)
                chat_manager.add_message("assistant", response)
    
    # Tab 2: Dashboard
    with tabs[1]:
        st.markdown("### Dashboard de Análisis")
        
        df = st.session_state.df
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-label">VTR Promedio</div>
                <div class="metric-value">{:.4f}</div>
            </div>
            """.format(df['VTR'].mean()), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Inversión Total</div>
                <div class="metric-value">${:,.0f}</div>
            </div>
            """.format(df['Cost'].sum()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Videos Únicos</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(df['Enlace video'].nunique()), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Impresiones</div>
                <div class="metric-value">{:,.0f}</div>
            </div>
            """.format(df['Impressions'].sum()), unsafe_allow_html=True)
        
        # Visualizaciones
        st.markdown("### Análisis Visual")
        create_dashboard_visualizations(df)
    
    # Tab 3: Predicción
    with tabs[2]:
        st.markdown("### Predictor de VTR")
        
        if 'ml_pipeline' in st.session_state:
            df = st.session_state.df
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    territorio = st.selectbox(
                        "Territorio",
                        options=df['Territorio'].dropna().unique()
                    )
                    sexo = st.selectbox(
                        "Sexo",
                        options=df['Sexo'].dropna().unique()
                    )
                    edad = st.selectbox(
                        "Edad",
                        options=df['Edad'].dropna().unique()
                    )
                
                with col2:
                    tematica = st.selectbox(
                        "Temática",
                        options=df['Temática'].dropna().unique()
                    )
                    presupuesto = st.number_input(
                        "Presupuesto (COP)",
                        min_value=100000,
                        value=1000000,
                        step=50000
                    )
                
                submitted = st.form_submit_button(
                    "Predecir VTR",
                    type="primary",
                    use_container_width=True
                )
                
                if submitted:
                    # Crear dataframe de entrada
                    input_data = pd.DataFrame([{
                        'Territorio': territorio,
                        'Sexo': sexo,
                        'Edad': edad,
                        'Temática': tematica,
                        'Cost': presupuesto
                    }])
                    
                    # Predecir
                    try:
                        prediction = st.session_state.ml_pipeline.best_model.predict(input_data)[0]
                        
                        # Mostrar resultado
                        st.success(f"VTR Predicho: **{prediction:.4f}**")
                        
                        # Comparar con promedio
                        avg_vtr = df['VTR'].mean()
                        diff_pct = ((prediction - avg_vtr) / avg_vtr) * 100
                        
                        if diff_pct > 0:
                            st.info(f"Este VTR está {diff_pct:.1f}% por encima del promedio")
                        else:
                            st.warning(f"Este VTR está {abs(diff_pct):.1f}% por debajo del promedio")
                            
                    except Exception as e:
                        st.error(f"Error en predicción: {e}")
        else:
            st.info("Por favor, carga los datos primero desde el panel lateral")
    
    # Tab 4: Resultados ML
    with tabs[3]:
        st.markdown("### Resultados del Análisis de Machine Learning")
        
        if 'ml_pipeline' in st.session_state:
            results_df = st.session_state.ml_pipeline.results_df
            
            st.markdown(f"#### Mejor Modelo: **{st.session_state.ml_pipeline.best_model_name}**")
            
            # Tabla de resultados
            st.dataframe(
                results_df.style.highlight_max(subset=['R2'], color='lightgreen'),
                use_container_width=True
            )
            
            # Gráfico de comparación
            fig = px.bar(
                results_df.reset_index(),
                x='index',
                y='R2',
                title='Comparación de Modelos (R²)',
                labels={'index': 'Modelo', 'R2': 'R² Score'},
                color='R2',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Por favor, carga los datos primero desde el panel lateral")

else:
    # Pantalla de bienvenida
    ytdlp_status = '✓ Instalado' if YTDLP_AVAILABLE else '✗ No instalado'
    cv2_status = '✓ Instalado' if CV2_AVAILABLE else '✗ No instalado'
    moviepy_status = '✓ Instalado' if MOVIEPY_AVAILABLE else '✗ No instalado'
    whisper_status = '✓ Instalado' if WHISPER_AVAILABLE else '✗ No instalado'
    sr_status = '✓ Instalado' if SPEECH_RECOGNITION_AVAILABLE else '✗ No instalado'
    
    st.markdown(f"""
    <div class="analysis-card fade-in">
        <h2>Bienvenido al Asistente Azembi</h2>
        <p>Sistema profesional de análisis de campañas con inteligencia artificial.</p>
        
        <h3>Características principales:</h3>
        <ul>
            <li><strong>Análisis completo de videos de YouTube:</strong> Extrae transcripción, analiza frames y predice VTR</li>
            <li><strong>Predicción de VTR con IA:</strong> Estima el rendimiento esperado basándose en datos históricos</li>
            <li><strong>Chat inteligente:</strong> Múltiples sesiones con contexto persistente</li>
            <li><strong>Machine Learning avanzado:</strong> 5 modelos diferentes para predicción precisa</li>
            <li><strong>Dashboard interactivo:</strong> Visualizaciones en tiempo real de métricas clave</li>
        </ul>
        
        <h3>Cómo analizar un video de YouTube:</h3>
        <ol>
            <li>Carga los datos haciendo clic en "Cargar Datos" en el panel lateral</li>
            <li>Ve a la pestaña "Chat Inteligente"</li>
            <li>Pega el enlace del video de YouTube (ej: https://www.youtube.com/watch?v=VVxA52mxcXA)</li>
            <li>El sistema analizará automáticamente el video y predecirá su VTR</li>
        </ol>
        
        <h3>Librerías opcionales para análisis completo de video:</h3>
        <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
# Instalar todas las dependencias opcionales:
pip install yt-dlp moviepy opencv-python-headless
pip install SpeechRecognition openai-whisper Pillow
        </pre>
        
        <p><strong>Estado actual de librerías:</strong></p>
        <ul>
            <li>yt-dlp: {ytdlp_status} (Requerido para descargar videos)</li>
            <li>OpenCV: {cv2_status} (Requerido para análisis visual)</li>
            <li>MoviePy: {moviepy_status} (Requerido para extraer audio)</li>
            <li>Whisper: {whisper_status} (Opcional - transcripción avanzada)</li>
            <li>Speech Recognition: {sr_status} (Opcional - transcripción básica)</li>
            <li>AWS Bedrock: {'✓ Configurado' if BEDROCK_AVAILABLE else '✗ No configurado'} (Requerido para IA)</li>
        </ul>
        
        <p><strong>Para comenzar:</strong> Haz clic en "Cargar Datos" en el panel lateral.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6C757D;">
    <p>Asistente Azembi - Sistema Profesional de Análisis</p>
</div>
""", unsafe_allow_html=True)