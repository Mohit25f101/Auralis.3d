# ==============================
# 📄 main.py - AURALIS v5.0 STANDALONE
# ==============================
# COMPLETE STANDALONE VERSION — NO EXTERNAL LOCAL IMPORTS
# Rich JSON output: Emotions, Speaker Confidence, Quality Score,
# Semantic Analysis, Entity Extraction, Evidence, and more.
# ==============================

import os
import sys
import logging
import warnings

# ── Silence everything ─────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
# ==============================
# 📦 IMPORTS
# ==============================
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
from datetime import datetime
from collections import defaultdict
import numpy as np
import csv
import requests
import shutil
import json
import uuid
import re
import time
import subprocess
import tempfile

# ==============================
# 📦 ML IMPORTS
# ==============================
print("\n" + "="*60)
print("🔄 Loading ML Libraries...")
print("="*60)

tf = None
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    print(f"   ✅ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"   ⚠️ TensorFlow: {e}")

hub = None
try:
    import tensorflow_hub as hub
    print("   ✅ TensorFlow Hub")
except Exception as e:
    print(f"   ⚠️ TensorFlow Hub: {e}")

librosa = None
try:
    import librosa
    print(f"   ✅ Librosa {librosa.__version__}")
except Exception as e:
    print(f"   ⚠️ Librosa: {e}")

sf = None
try:
    import soundfile as sf
    print("   ✅ Soundfile")
except:
    pass

wavfile = None
try:
    from scipy.io import wavfile
    print("   ✅ Scipy")
except:
    pass

pipeline_func = None
try:
    from transformers import pipeline as pipeline_func
    print("   ✅ Transformers")
except Exception as e:
    print(f"   ⚠️ Transformers: {e}")

torch = None
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
except:
    pass

# deep_translator for non-English translation
deep_translator_available = False
try:
    from deep_translator import GoogleTranslator
    deep_translator_available = True
    print("   ✅ deep_translator")
except ImportError:
    print("   ⚠️ deep_translator not installed")
    print("      Install: pip install deep-translator")
except Exception as e:
    print(f"   ⚠️ deep_translator error: {e}")

print("="*60)


# ==============================
# ⚙️ CONFIGURATION
# ==============================
SAMPLE_RATE = 16000
MAX_DURATION = 60.0
MIN_DURATION = 0.5

LOCATIONS = [
    "Airport Terminal", "Railway Station", "Bus Terminal", "Metro/Subway",
    "Hospital", "Shopping Mall", "Office Building", "School/University",
    "Restaurant/Cafe", "Street/Road", "Home/Residential", "Park/Outdoor",
    "Stadium/Arena", "Parking Area", "Construction Site", "Factory/Industrial",
    "Religious Place", "Government Office", "Bank", "Hotel/Lodge",
    "Cinema/Theater", "Gym/Sports Center", "Market/Bazaar", "Unknown"
]

SITUATIONS = [
    "Normal/Quiet", "Busy/Crowded", "Emergency", "Boarding/Departure",
    "Waiting", "Traffic", "Meeting/Conference", "Announcement",
    "Celebration/Event", "Construction", "Weather Event", "Accident",
    "Medical Emergency", "Security Alert", "Rush Hour", "Flight Delay",
    "Train Delay", "Sports Event", "Concert/Music", "Unknown"
]

# Auto-detect FFmpeg path based on OS
if os.name == 'nt':  # Windows
    FFMPEG_PATH = r"D:\DD\ffmpeg-2026-01-07-git-af6a1dd0b2-full_build\ffmpeg-2026-01-07-git-af6a1dd0b2-full_build\bin"
else:  # Linux/Mac
    FFMPEG_PATH = "/usr/bin"
LEARNING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_data.json")


# ==============================
# 🎬 FFMPEG
# ==============================
def setup_ffmpeg():
    """Setup FFmpeg with comprehensive path checking for both Windows and Linux."""
    try:
        # Add common Linux paths to environment PATH first
        if os.name != 'nt':  # Linux/Mac
            common_paths = ["/usr/bin", "/usr/local/bin", "/bin", "/usr/sbin", "/sbin"]
            current_path = os.environ.get("PATH", "")
            
            for path in common_paths:
                if path not in current_path:
                    os.environ["PATH"] = path + os.pathsep + current_path
                    current_path = os.environ["PATH"]
        
        # Now check if ffmpeg is available
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print(f"✅ FFmpeg found at: {ffmpeg_path}")
            return True
        
        # Try custom FFMPEG_PATH as fallback (for Windows or custom installations)
        if 'FFMPEG_PATH' in globals() and os.path.isdir(FFMPEG_PATH):
            os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                print(f"✅ FFmpeg enabled from custom path: {ffmpeg_path}")
                return True
        
        # Last resort: try direct paths
        if os.name != 'nt':
            for direct_path in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
                if os.path.exists(direct_path):
                    print(f"✅ FFmpeg found at: {direct_path}")
                    return True
        
        print("⚠️ FFmpeg not found - audio conversion will be limited")
        print("   Install FFmpeg: sudo apt-get install ffmpeg (Linux)")
        return False
    except Exception as e:
        print(f"⚠️ FFmpeg setup error: {e}")
        return False


def convert_audio(input_path, output_path):
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
        subprocess.run(cmd, capture_output=True, timeout=60)
        return os.path.exists(output_path)
    except:
        return False


# ==============================
# 📚 LEARNING SYSTEM
# ==============================
class LearningSystem:
    def __init__(self):
        self.boosts = defaultdict(lambda: defaultdict(float))
        self.corrections = 0
        self._load()

    def _load(self):
        try:
            if os.path.exists(LEARNING_FILE):
                with open(LEARNING_FILE, 'r') as f:
                    data = json.load(f)
                    self.boosts = defaultdict(lambda: defaultdict(float),
                        {k: defaultdict(float, v) for k, v in data.get('boosts', {}).items()})
                    self.corrections = data.get('corrections', 0)
                if self.corrections > 0:
                    print(f"📚 Loaded {self.corrections} learned corrections")
        except:
            pass

    def save(self):
        try:
            with open(LEARNING_FILE, 'w') as f:
                json.dump({
                    'boosts': {k: dict(v) for k, v in self.boosts.items()},
                    'corrections': self.corrections
                }, f)
        except:
            pass

    def learn(self, category, value, text, sounds):
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        for w in words:
            self.boosts[category][f"{value}::{w}"] += 0.1
        for s in sounds[:5]:
            self.boosts[category][f"{value}::sound::{s.lower()}"] += 0.12
        self.corrections += 1
        if self.corrections % 3 == 0:
            self.save()

    def get_boost(self, category, value, text, sounds):
        boost = 0.0
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        for w in words:
            boost += self.boosts[category].get(f"{value}::{w}", 0)
        for s in sounds:
            boost += self.boosts[category].get(f"{value}::sound::{s.lower()}", 0)
        return min(boost, 0.3)


# ==============================
# 🔊 AUDIO LOADER
# ==============================
class AudioLoader:
    def __init__(self):
        self.sr = SAMPLE_RATE

    def load(self, path):
        if not os.path.exists(path):
            raise Exception("File not found")
        audio = None

        if librosa and audio is None:
            try:
                audio, _ = librosa.load(path, sr=self.sr, mono=True)
                print("   ✅ Loaded with Librosa")
            except:
                pass

        if sf and audio is None:
            try:
                data, orig_sr = sf.read(path)
                audio = data.astype(np.float32)
                if orig_sr != self.sr and librosa:
                    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                print("   ✅ Loaded with Soundfile")
            except:
                pass

        if audio is None:
            temp = path + ".converted.wav"
            try:
                if convert_audio(path, temp) and wavfile:
                    orig_sr, data = wavfile.read(temp)
                    if data.dtype == np.int16:
                        audio = data.astype(np.float32) / 32768.0
                    elif data.dtype == np.float32:
                        audio = data
                    else:
                        audio = data.astype(np.float32)
                    if orig_sr != self.sr and librosa:
                        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)
                    print("   ✅ Loaded with FFmpeg")
            except Exception as e:
                print(f"   ⚠️ FFmpeg error: {e}")
            finally:
                if os.path.exists(temp):
                    os.remove(temp)

        if audio is None:
            raise Exception("Could not load audio")

        audio = audio.astype(np.float32)
        audio = audio - np.mean(audio)
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx * 0.95
        return audio, len(audio) / self.sr

    def save_wav(self, audio, path):
        if wavfile:
            wavfile.write(path, self.sr, (audio * 32767).astype(np.int16))
        return path


# ==============================
# 🗣️ WHISPER MANAGER  (translate mode → always returns English)
# ==============================
class WhisperManager:
    HALLUCINATIONS = [
        "thank you for watching", "please subscribe", "like and subscribe",
        "thanks for watching", "bye bye", "goodbye", "[music]", "[applause]",
    ]

    def __init__(self):
        self.pipe = None
        self.pipe_transcribe = None   # separate pipeline for native-language transcription
        self.loaded = False

    def load(self):
        if self.loaded:
            return True
        if not pipeline_func or not torch:
            print("❌ Whisper: Transformers or PyTorch not available")
            return False
        try:
            print("🔄 Loading Whisper...")
            # Disable torchcodec to avoid errors
            import os
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
            
            self.pipe = pipeline_func(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                chunk_length_s=30,
                device="cpu",  # Explicitly use CPU to avoid device issues
                return_timestamps=False,  # Disable timestamps to avoid torchcodec
            )
            self.loaded = True
            print("✅ Whisper loaded!")
            return True
        except Exception as e:
            print(f"❌ Whisper error: {e}")
            # Fallback: try without chunk_length_s
            try:
                self.pipe = pipeline_func(
                    "automatic-speech-recognition",
                    model="openai/whisper-small",
                    device="cpu",
                )
                self.loaded = True
                print("✅ Whisper loaded (fallback mode)!")
                return True
            except Exception as e2:
                print(f"❌ Whisper fallback failed: {e2}")
                return False

    def transcribe(self, path):
        """Returns translated (English) text + detected language + native transcription."""
        if not self.loaded:
            return {"text": "", "confidence": 0, "is_reliable": False,
                    "detected_language": "unknown", "native_text": ""}

        try:
            # 1) Translate mode → English
            result_en = self.pipe(path, generate_kwargs={"task": "translate"})
            raw_en = result_en.get("text", "").strip()

            # 2) Transcribe mode → native language text + language detection
            result_native = self.pipe(path, generate_kwargs={"task": "transcribe"})
            raw_native = result_native.get("text", "").strip()

            # Whisper doesn't expose language directly in HF pipeline easily,
            # so we detect from native text using our LanguageDetector later.
            # For now store raw_native.

            # ── Hallucination / repetition guard on English ──
            if not raw_en or len(raw_en) < 3:
                return {"text": "", "native_text": raw_native, "confidence": 0.2,
                        "is_reliable": False, "detected_language": "unknown"}

            lower = raw_en.lower()
            for h in self.HALLUCINATIONS:
                if h in lower:
                    return {"text": "", "raw": raw_en, "native_text": raw_native,
                            "confidence": 0.2, "is_reliable": False, "detected_language": "unknown"}

            words = raw_en.split()
            if len(words) > 5 and len(set(words)) / len(words) < 0.35:
                return {"text": "", "raw": raw_en, "native_text": raw_native,
                        "confidence": 0.25, "is_reliable": False, "detected_language": "unknown"}

            clean = ' '.join(raw_en.split())
            clean = re.sub(r'\[.*?\]', '', clean).strip()

            conf = 0.90 if len(words) >= 10 else (0.85 if len(words) >= 5 else 0.75)

            # Try to extract language from result if available
            detected_lang = "unknown"
            if hasattr(result_native, 'get') and 'chunks' in result_native:
                # Try to get language from chunks
                for chunk in result_native.get('chunks', []):
                    if 'language' in chunk:
                        detected_lang = chunk['language']
                        break
            
            return {
                "text": clean,
                "raw": raw_en,
                "native_text": raw_native,
                "confidence": conf,
                "is_reliable": True,
                "detected_language": detected_lang
            }

        except Exception as e:
            error_msg = str(e)
            # Filter out verbose torchcodec errors
            if "torchcodec" in error_msg or "libtorchcodec" in error_msg:
                print(f"   ⚠️ Transcription warning: Audio codec compatibility issue (continuing anyway)")
            else:
                print(f"   ⚠️ Transcription error: {error_msg[:80]}")
            
            return {"text": "", "confidence": 0, "is_reliable": False,
                    "detected_language": "unknown", "native_text": "", "error": error_msg[:100]}


# ==============================
# 🔊 YAMNET MANAGER
# ==============================
class YAMNetManager:
    FILTER_OUT = {"silence", "white noise", "pink noise", "static"}
    GENERIC = {
        "speech", "narration", "monologue", "conversation",
        "male speech", "female speech", "inside", "outside", "room",
    }
    SOUND_LOCATIONS = {
        "aircraft": ["Airport Terminal"], "airplane": ["Airport Terminal"],
        "jet engine": ["Airport Terminal"], "helicopter": ["Airport Terminal"],
        "train": ["Railway Station"], "railroad": ["Railway Station"],
        "rail transport": ["Railway Station"], "subway": ["Metro/Subway"],
        "car": ["Street/Road"], "vehicle": ["Street/Road"],
        "traffic": ["Street/Road"], "horn": ["Street/Road"],
        "siren": ["Hospital", "Street/Road"], "ambulance": ["Hospital"],
        "crowd": ["Airport Terminal", "Railway Station", "Shopping Mall", "Stadium/Arena"],
        "applause": ["Stadium/Arena"], "cheering": ["Stadium/Arena"],
        "bell": ["Religious Place"], "church bell": ["Religious Place"],
        "construction": ["Construction Site"], "hammer": ["Construction Site"],
        "bird": ["Park/Outdoor"], "water": ["Park/Outdoor"],
    }
    SOUND_SITUATIONS = {
        "siren": ["Emergency"], "alarm": ["Emergency"],
        "ambulance": ["Medical Emergency"], "crowd": ["Busy/Crowded"],
        "applause": ["Celebration/Event"], "cheering": ["Sports Event"],
        "traffic": ["Traffic"], "horn": ["Traffic"],
        "construction": ["Construction"], "thunder": ["Weather Event"],
    }

    def __init__(self):
        self.model = None
        self.labels = []
        self.loaded = False

    def load(self):
        if self.loaded:
            return True
        if not hub or not tf:
            print("❌ YAMNet: TensorFlow not available")
            return False
        try:
            print("🔄 Loading YAMNet...")
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            self._load_labels()
            self.loaded = True
            print(f"✅ YAMNet loaded ({len(self.labels)} classes)")
            return True
        except Exception as e:
            print(f"❌ YAMNet error: {e}")
            return False

    def _load_labels(self):
        try:
            url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
            r = requests.get(url, timeout=10)
            reader = csv.reader(r.text.splitlines())
            next(reader)
            self.labels = [row[2] for row in reader]
        except:
            self.labels = [f"Sound_{i}" for i in range(521)]

    def analyze(self, audio):
        if not self.loaded:
            return {"sounds": {}, "location_hints": [], "situation_hints": []}
        try:
            scores, _, _ = self.model(audio.astype(np.float32))
            mean = tf.reduce_mean(scores, axis=0).numpy()
            all_sounds = {}
            for i in range(len(mean)):
                if mean[i] > 0.01 and i < len(self.labels):
                    all_sounds[self.labels[i]] = float(mean[i])
            filtered = self._filter(all_sounds)
            return {
                "sounds": filtered,
                "all": all_sounds,
                "location_hints": self._location_hints(all_sounds),
                "situation_hints": self._situation_hints(all_sounds),
            }
        except Exception as e:
            print(f"   ⚠️ YAMNet error: {e}")
            return {"sounds": {}, "location_hints": [], "situation_hints": []}

    def _filter(self, sounds):
        result = {}
        generic = {}
        for s, score in sounds.items():
            lower = s.lower()
            if any(f in lower for f in self.FILTER_OUT):
                continue
            if any(g in lower for g in self.GENERIC):
                if score > 0.05:
                    generic[s] = score
            elif score > 0.02:
                result[s] = round(score, 3)
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:12])
        if not result and generic:
            result = dict(sorted(generic.items(), key=lambda x: x[1], reverse=True)[:5])
        return result

    def _location_hints(self, sounds):
        hints = defaultdict(float)
        for s, score in sounds.items():
            lower = s.lower()
            for kw, locs in self.SOUND_LOCATIONS.items():
                if kw in lower:
                    for loc in locs:
                        hints[loc] = max(hints[loc], score)
        return sorted(hints.items(), key=lambda x: x[1], reverse=True)[:5]

    def _situation_hints(self, sounds):
        hints = defaultdict(float)
        for s, score in sounds.items():
            lower = s.lower()
            for kw, sits in self.SOUND_SITUATIONS.items():
                if kw in lower:
                    for sit in sits:
                        hints[sit] = max(hints[sit], score)
        return sorted(hints.items(), key=lambda x: x[1], reverse=True)[:5]


# ==============================
# 🌍 LANGUAGE DETECTOR  (inline, pure-Python, 60+ languages)
# ==============================
class LanguageDetector:
    LANGUAGES = {
        "en": {"name": "English", "native": "English"},
        "zh": {"name": "Chinese", "native": "中文"},
        "hi": {"name": "Hindi", "native": "हिन्दी"},
        "ar": {"name": "Arabic", "native": "العربية"},
        "es": {"name": "Spanish", "native": "Español"},
        "fr": {"name": "French", "native": "Français"},
        "de": {"name": "German", "native": "Deutsch"},
        "ja": {"name": "Japanese", "native": "日本語"},
        "ko": {"name": "Korean", "native": "한국어"},
        "ru": {"name": "Russian", "native": "Русский"},
        "pt": {"name": "Portuguese", "native": "Português"},
        "it": {"name": "Italian", "native": "Italiano"},
        "nl": {"name": "Dutch", "native": "Nederlands"},
        "ta": {"name": "Tamil", "native": "தமிழ்"},
        "te": {"name": "Telugu", "native": "తెలుగు"},
        "bn": {"name": "Bengali", "native": "বাংলা"},
        "th": {"name": "Thai", "native": "ไทย"},
        "vi": {"name": "Vietnamese", "native": "Tiếng Việt"},
        "id": {"name": "Indonesian", "native": "Bahasa Indonesia"},
        "tr": {"name": "Turkish", "native": "Türkçe"},
        "pl": {"name": "Polish", "native": "Polski"},
        "uk": {"name": "Ukrainian", "native": "Українська"},
        "fa": {"name": "Persian", "native": "فارسی"},
        "ur": {"name": "Urdu", "native": "اردو"},
        "gu": {"name": "Gujarati", "native": "ગુજરાતી"},
        "mr": {"name": "Marathi", "native": "मराठी"},
        "kn": {"name": "Kannada", "native": "ಕನ್ನಡ"},
        "ml": {"name": "Malayalam", "native": "മലയാളം"},
        "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ"},
    }

    # Script → language code mapping
    SCRIPT_PATTERNS = {
        "devanagari": (r'[\u0900-\u097F]', ["hi", "mr"]),
        "bengali":    (r'[\u0980-\u09FF]', ["bn"]),
        "tamil":      (r'[\u0B80-\u0BFF]', ["ta"]),
        "telugu":     (r'[\u0C00-\u0C7F]', ["te"]),
        "kannada":    (r'[\u0C80-\u0CFF]', ["kn"]),
        "malayalam":  (r'[\u0D00-\u0D7F]', ["ml"]),
        "gujarati":   (r'[\u0A80-\u0AFF]', ["gu"]),
        "gurmukhi":   (r'[\u0A00-\u0A7F]', ["pa"]),
        "thai":       (r'[\u0E00-\u0E7F]', ["th"]),
        "arabic":     (r'[\u0600-\u06FF\u0750-\u077F]', ["ar", "fa", "ur"]),
        "chinese":    (r'[\u4E00-\u9FFF\u3400-\u4DBF]', ["zh"]),
        "japanese_h": (r'[\u3040-\u309F]', ["ja"]),
        "japanese_k": (r'[\u30A0-\u30FF]', ["ja"]),
        "korean":     (r'[\uAC00-\uD7AF\u1100-\u11FF]', ["ko"]),
        "cyrillic":   (r'[\u0400-\u04FF]', ["ru", "uk"]),
    }

    MARKERS = {
        "en": ["the","and","is","are","was","were","have","will","this","that","with","from","for","not","what","when","there"],
        "es": ["que","de","en","el","la","los","las","por","con","para","una","como","más","pero"],
        "fr": ["de","la","le","les","et","en","un","une","du","que","est","dans","qui","pour"],
        "de": ["der","die","und","in","den","von","zu","das","mit","sich","des","auf","für","ist"],
        "it": ["di","che","la","il","un","una","per","non","sono","da","con","si","come","anche"],
        "pt": ["que","de","em","um","uma","para","com","não","por","mais","como","mas","foi"],
        "zh": ["的","是","在","不","了","有","和","人","这","中","大","为","上","个"],
        "ja": ["の","に","は","を","た","が","で","て","と","し","れ","さ"],
        "ko": ["이","는","의","을","에","가","를","으로","하","고"],
        "ru": ["и","в","не","на","что","я","с","он","как","это","все","она"],
        "hi": ["का","की","के","में","है","और","को","से","पर","यह"],
        "ar": ["في","من","على","إلى","أن","هذا","التي","الذي","مع","كان"],
        "tr": ["bir","ve","bu","için","olan","ile","de","da","olarak","gibi"],
        "vi": ["của","và","các","là","trong","được","có","này","cho","với"],
        "th": ["ที่","และ","ใน","ของ","เป็น","ได้","มี","การ","จะ","ไม่"],
        "id": ["yang","dan","di","ini","dari","untuk","dengan","tidak","adalah","ke"],
    }

    def detect(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 2:
            return {"code": "unknown", "name": "Unknown", "native": "Unknown", "confidence": 0.0}

        # 1) Script-based detection (best for non-Latin)
        for script, (pattern, langs) in self.SCRIPT_PATTERNS.items():
            if re.search(pattern, text):
                code = langs[0]
                info = self.LANGUAGES.get(code, {})
                return {"code": code, "name": info.get("name", "Unknown"),
                        "native": info.get("native", "Unknown"), "confidence": 0.92}

        # 2) Word-marker detection (for Latin-script languages)
        words = set(re.findall(r'\b\w+\b', text.lower()))
        best_lang, best_score = "en", 0.0
        for lang, markers in self.MARKERS.items():
            hits = sum(1 for m in markers if m in words)
            score = hits / len(markers)
            if score > best_score:
                best_lang, best_score = lang, score

        info = self.LANGUAGES.get(best_lang, {"name": "English", "native": "English"})
        conf = min(0.5 + best_score * 1.2, 0.95) if best_score > 0 else 0.6
        return {"code": best_lang, "name": info.get("name", "Unknown"),
                "native": info.get("native", "Unknown"), "confidence": round(conf, 3)}


# ==============================
# 😊 EMOTION DETECTOR  (inline, pure-Python)
# ==============================
class EmotionDetector:
    EMOTION_KEYWORDS = {
        "joy":       ["happy","glad","excited","wonderful","great","amazing","fantastic","excellent","delighted","thrilled","ecstatic","cheerful","joyful"],
        "sadness":   ["sad","sorry","unfortunately","regret","disappoint","unhappy","depressed","upset","miserable","grief","sorrow","gloomy"],
        "anger":     ["angry","furious","annoyed","frustrated","irritated","outraged","mad","livid","enraged","hostile","resentful","irate"],
        "fear":      ["afraid","scared","worried","anxious","nervous","terrified","frightened","panicked","alarmed","concerned","dread"],
        "surprise":  ["surprised","amazed","astonished","shocked","unexpected","startled","stunned","bewildered","astounded"],
        "neutral":   ["attention","please","note","information","announce","inform","notice","update","regarding"],
        "urgency":   ["urgent","immediately","emergency","now","hurry","critical","asap","right away","at once","quickly","rush","priority"],
        "calm":      ["calm","peaceful","relaxed","gentle","quiet","serene","tranquil","soothing","steady","patient"],
    }

    SOUND_EMOTIONS = {
        "music": {"calm": 0.5, "joy": 0.3},
        "alarm": {"urgency": 0.8, "fear": 0.4},
        "siren": {"urgency": 0.9, "fear": 0.5},
        "crying": {"sadness": 0.8},
        "laughter": {"joy": 0.9},
        "applause": {"joy": 0.7, "surprise": 0.3},
        "crowd": {"neutral": 0.5},
        "yelling": {"anger": 0.6, "urgency": 0.5},
        "screaming": {"fear": 0.7, "urgency": 0.8},
    }

    def analyze(self, audio: np.ndarray, text: str, sounds: Dict[str, float]) -> Dict[str, Any]:
        text_lower = text.lower()
        text_emo = self._text_emotions(text_lower)
        audio_emo = self._audio_emotions(audio)
        sound_emo = self._sound_emotions(sounds)
        micro = self._micro_emotions(text)

        # Weighted merge
        combined = {}
        for emo in set(list(text_emo) + list(audio_emo) + list(sound_emo)):
            combined[emo] = text_emo.get(emo, 0)*0.50 + audio_emo.get(emo, 0)*0.30 + sound_emo.get(emo, 0)*0.20

        primary = max(combined.items(), key=lambda x: x[1]) if combined else ("neutral", 0.5)
        valence, arousal = self._valence_arousal(combined)
        intensity = self._intensity(audio, text)

        # Build micro_emotions summary for JSON
        micro_summary = {}
        for m in micro:
            micro_summary[m["type"]] = m["score"]

        return {
            "primary_emotion": primary[0],
            "intensity": round(primary[1], 3),
            "micro_emotions": micro_summary,
            "all_emotions": {k: round(v, 3) for k, v in sorted(combined.items(), key=lambda x: -x[1]) if v > 0.05},
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "emotional_intensity": round(intensity, 3),
        }

    def _text_emotions(self, text):
        emo = {}
        for name, kws in self.EMOTION_KEYWORDS.items():
            score = sum(0.22 for k in kws if k in text)
            if score > 0:
                emo[name] = min(score, 1.0)
        return emo

    def _audio_emotions(self, audio):
        emo = {}
        energy = float(np.mean(np.abs(audio)))
        energy_var = float(np.var(np.abs(audio)))
        if energy > 0.1 and energy_var > 0.01:
            emo["urgency"] = min(energy * 3, 0.8)
            emo["anger"] = min(energy * 2, 0.6)
        if energy < 0.05:
            emo["calm"] = 0.5
            emo["sadness"] = 0.3
        if 0.05 <= energy <= 0.1:
            emo["neutral"] = 0.5
        return emo

    def _sound_emotions(self, sounds):
        emo = {}
        for s, conf in sounds.items():
            key = s.lower()
            for se_key, mapping in self.SOUND_EMOTIONS.items():
                if se_key in key:
                    for e, base in mapping.items():
                        emo[e] = max(emo.get(e, 0), base * conf)
        return emo

    def _micro_emotions(self, text):
        micro = []
        patterns = {
            "urgency":    (r'\b(urgent|immediately|emergency|now|hurry|asap)\b', 0.8),
            "confidence": (r'\b(definitely|certainly|absolutely|sure|confident|must|will)\b', 0.7),
            "hesitation": (r'\b(um|uh|er|hmm|maybe|perhaps|might|possibly|probably)\b', 0.5),
            "politeness": (r'\b(please|thank|kindly|appreciate)\b', 0.6),
            "formality":  (r'\b(ladies|gentlemen|sir|madam|dear|respected)\b', 0.7),
        }
        for name, (pat, base_score) in patterns.items():
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                micro.append({"type": name, "score": round(min(base_score + len(matches)*0.1, 1.0), 2)})
        return micro

    def _valence_arousal(self, emo):
        V = {"joy": 1.0, "calm": 0.5, "surprise": 0.3, "neutral": 0.0, "sadness": -0.7, "fear": -0.8, "anger": -0.6}
        A = {"joy": 0.7, "surprise": 0.8, "anger": 0.9, "fear": 0.8, "urgency": 0.9, "calm": 0.2, "sadness": 0.3, "neutral": 0.4}
        total = sum(emo.values()) or 1
        v = sum(V.get(e, 0)*s for e, s in emo.items()) / total
        a = sum(A.get(e, 0.5)*s for e, s in emo.items()) / total
        return max(-1, min(1, v)), max(0, min(1, a))

    def _intensity(self, audio, text):
        ai = min(float(np.mean(np.abs(audio))) * 10, 1.0)
        ti = min(text.count("!") * 0.1 + sum(1 for c in text if c.isupper()) / (len(text) or 1), 1.0)
        return ai * 0.6 + ti * 0.4


# ==============================
# 🎤 SPEAKER CONFIDENCE SCORER  (inline, pure-Python)
# ==============================
class ConfidenceScorer:
    CONFIDENT_WORDS = ["definitely","certainly","absolutely","sure","confident","guarantee","must","will","know","clear","obvious","fact","proven"]
    HESITATION_WORDS = ["maybe","perhaps","possibly","might","could be","not sure","i think","i guess","probably","seems","um","uh","er","like","sort of","kind of"]

    def score(self, audio: np.ndarray, text: str, trans_conf: float) -> Dict[str, Any]:
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words) or 1

        # Linguistic
        conf_hits = sum(1 for w in self.CONFIDENT_WORDS if w in text_lower)
        hes_hits  = sum(1 for w in self.HESITATION_WORDS if w in text_lower)
        linguistic = 0.5 + (conf_hits - hes_hits) * 0.08
        linguistic = max(0.1, min(0.95, linguistic))

        # Prosodic (from audio)
        energy = float(np.mean(np.abs(audio)))
        energy_var = float(np.var(np.abs(audio)))
        # Stable, moderate energy = confident
        prosodic = 0.5
        if 0.04 < energy < 0.15:
            prosodic += 0.15
        if energy_var < 0.01:
            prosodic += 0.1
        prosodic = max(0.1, min(0.95, prosodic))

        # Structure (complete sentences, declarative)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        complete = sum(1 for s in sentences if len(s.split()) >= 3)
        structure = 0.5 + (complete / max(len(sentences), 1) - 0.5) * 0.2
        structure = max(0.1, min(0.95, structure))

        # Vocabulary (unique ratio + avg word length)
        unique_ratio = len(set(words)) / word_count
        avg_len = sum(len(w) for w in words) / word_count if words else 4
        vocabulary = 0.3 + unique_ratio * 0.35 + min(avg_len / 6, 1.0) * 0.35
        vocabulary = max(0.1, min(0.95, vocabulary))

        # Weighted combo → scale 1-10
        raw = linguistic*0.35 + prosodic*0.25 + structure*0.20 + vocabulary*0.20
        final_score = round(1 + raw * 9, 1)
        final_score = max(1.0, min(10.0, final_score))

        level = "Very High" if final_score >= 8.5 else ("High" if final_score >= 7 else ("Medium" if final_score >= 5 else ("Low" if final_score >= 3 else "Very Low")))

        # Breakdown for JSON
        breakdown = {
            "fluency": round(linguistic * 10, 1),
            "volume_stability": round(prosodic * 10, 1),
            "sentence_structure": round(structure * 10, 1),
            "vocabulary_richness": round(vocabulary * 10, 1),
        }

        return {
            "score": final_score,
            "level": level,
            "breakdown": breakdown,
            "confidence_words_found": [w for w in self.CONFIDENT_WORDS if w in text_lower][:5],
            "hesitation_words_found": [w for w in self.HESITATION_WORDS if w in text_lower][:5],
        }


# ==============================
# 📊 QUALITY SCORER  (inline)
# ==============================
class QualityScorer:
    """Overall analysis quality score 1-10 with specific feedback."""

    def score(self, trans: Dict, sounds: Dict, location_conf: float, situation_conf: float) -> Dict[str, Any]:
        scores = []
        feedback = []

        # 1) Transcription quality
        if trans.get("is_reliable") and trans.get("text"):
            t = trans["text"]
            word_count = len(t.split())
            if word_count >= 10:
                scores.append(9)
            elif word_count >= 5:
                scores.append(7)
            else:
                scores.append(5)
                feedback.append("Short transcription — limited text analysis.")
        else:
            scores.append(3)
            feedback.append("Transcription unreliable — relying on sound analysis only.")

        # 2) Sound detection richness
        num_sounds = len(sounds.get("sounds", {}))
        if num_sounds >= 5:
            scores.append(9)
        elif num_sounds >= 2:
            scores.append(7)
        elif num_sounds >= 1:
            scores.append(5)
            feedback.append("Few sounds detected — analysis may be limited.")
        else:
            scores.append(2)
            feedback.append("No ambient sounds detected.")

        # 3) Location & situation confidence
        avg_conf = (location_conf + situation_conf) / 2
        if avg_conf >= 0.85:
            scores.append(9)
        elif avg_conf >= 0.70:
            scores.append(7)
        else:
            scores.append(5)
            feedback.append("Lower confidence in location/situation — consider providing longer audio.")

        # 4) Language detection
        if trans.get("detected_language") and trans["detected_language"] != "unknown":
            scores.append(8)
        else:
            scores.append(5)
            feedback.append("Language detection uncertain.")

        overall = round(sum(scores) / len(scores), 1)
        if not feedback:
            feedback.append("High-quality analysis with strong signals across all dimensions.")

        return {
            "score": overall,
            "max_score": 10,
            "feedback": feedback,
        }


# ==============================
# 🔍 ENTITY EXTRACTOR  (inline, pure-Python)
# ==============================
class EntityExtractor:
    def extract(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"entities": {}, "summary": "No text to extract entities from."}

        entities = {
            "numbers": [],
            "times": [],
            "locations": [],
            "durations": [],
            "money": [],
            "contact": [],
            "names": [],
        }

        # Numbers
        entities["numbers"] = list(set(re.findall(r'\b(?:number|no\.?|#)?\s*(\d{2,5})\b', text, re.IGNORECASE)))[:5]

        # Times
        entities["times"] = list(set(re.findall(
            r'\b(\d{1,2}[:.]\d{2}\s*(?:am|pm|AM|PM)?|\d{1,2}\s*(?:am|pm|AM|PM)|\d{4}\s*(?:hours|hrs))\b', text)))[:5]

        # Platform / Gate / Terminal
        plats = re.findall(r'(?:platform|gate|terminal|track)\s*(?:number|no\.?)?\s*(\d+)', text, re.IGNORECASE)
        if plats:
            entities["locations"].extend([f"Platform/Gate {p}" for p in plats])

        # City/station references
        city_refs = re.findall(r'\b(?:to|from|at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', text)
        entities["locations"].extend(city_refs[:3])
        entities["locations"] = list(set(entities["locations"]))[:5]

        # Durations
        entities["durations"] = list(set(re.findall(
            r'(\d+\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?))', text, re.IGNORECASE)))[:4]

        # Money
        entities["money"] = list(set(re.findall(
            r'(?:₹|rs\.?|inr|usd|\$|€|£)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)))[:3]

        # Contact / Phone
        entities["contact"] = list(set(re.findall(r'\b(\d{3}[-.]?\d{3}[-.]?\d{4}|\d{10})\b', text)))[:3]

        # Names (Title + Capitalized word)
        entities["names"] = list(set(re.findall(
            r'\b((?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', text)))[:3]

        # Summary line
        parts = []
        if entities["numbers"]:  parts.append(f"Numbers: {', '.join(entities['numbers'][:2])}")
        if entities["times"]:    parts.append(f"Times: {', '.join(entities['times'][:2])}")
        if entities["locations"]: parts.append(f"Locations: {', '.join(entities['locations'][:2])}")
        if entities["durations"]: parts.append(f"Durations: {', '.join(entities['durations'][:2])}")
        summary = " | ".join(parts) if parts else "No structured entities found."

        return {"entities": entities, "summary": summary}


# ==============================
# 🧠 SEMANTIC ANALYZER  (inline, pure-Python)
# ==============================
class SemanticAnalyzer:
    INTENT_PATTERNS = {
        "announcement": [
            r"attention\s+(?:please|all)", r"(?:ladies\s+and\s+)?gentlemen",
            r"(?:we\s+)?(?:would\s+like\s+to\s+)?(?:announce|inform)",
            r"(?:may\s+i\s+have\s+)?your\s+attention", r"passengers\s+are\s+(?:requested|informed)",
            r"kindly\s+note", r"please\s+be\s+(?:informed|advised)",
        ],
        "instruction": [
            r"please\s+(?:proceed|go|come|move|stand|wait|board|line\s+up)",
            r"(?:you\s+)?(?:must|should|need\s+to|have\s+to)",
            r"(?:do\s+not|don't)\s+\w+", r"(?:make\s+sure|ensure)\s+(?:to|that)",
            r"(?:keep|remain|stay)\s+\w+", r"step\s+(?:back|forward|aside)",
        ],
        "warning": [
            r"(?:warning|caution|danger|alert)", r"(?:be\s+)?careful",
            r"(?:emergency|urgent|immediately)", r"for\s+your\s+(?:safety|security)",
            r"(?:evacuate|leave|exit)\s+(?:now|immediately)?",
        ],
        "question": [
            r"\?$", r"^(?:what|where|when|why|how|who|which)",
            r"^(?:is|are|was|were|do|does|did|can|could|will|would|should)",
            r"(?:can|could)\s+you\s+(?:tell|help|show)",
        ],
        "request": [
            r"(?:please|kindly)\s+\w+", r"(?:can|could|would)\s+you\s+(?:please)?",
            r"(?:i\s+)?(?:need|want|would\s+like)", r"(?:help\s+me|assist\s+me)",
        ],
        "greeting": [
            r"(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))",
            r"(?:welcome\s+to|welcome\s+aboard)", r"(?:greetings|namaste)",
        ],
        "emergency": [
            r"(?:emergency|help|fire|accident)", r"(?:evacuate|escape)\s+(?:now|immediately)?",
            r"(?:call|dial)\s+(?:911|100|101|102|108|112)",
            r"(?:someone\s+)?(?:needs?\s+help|hurt|injured)",
        ],
        "information": [
            r"(?:the|this)\s+(?:train|flight|bus|metro)\s+(?:number|is)",
            r"(?:arriving|departing)\s+(?:at|from)", r"(?:scheduled|expected)\s+(?:at|for)",
        ],
    }

    def analyze(self, text: str, location: str, situation: str, sounds: List[str]) -> Dict[str, Any]:
        if not text:
            return self._empty()

        text_lower = text.lower().strip()

        # Intent
        intent, intent_conf = self._detect_intent(text_lower)

        # Entities (reuse EntityExtractor output passed in or re-extract)
        entities = EntityExtractor().extract(text)

        # Topics
        topics = self._extract_topics(text_lower, location, situation)

        # Action items
        actions = self._extract_actions(text)

        # Audience
        audience = self._detect_audience(text_lower)

        # Urgency
        urgency = self._assess_urgency(text_lower, intent)

        # Sentiment
        sentiment = self._sentiment(text_lower)

        # Summary
        summary = self._summary(text, intent, entities.get("entities", {}), audience)

        return {
            "intent": intent,
            "intent_confidence": round(intent_conf, 3),
            "summary": summary,
            "entities": entities.get("entities", {}),
            "topics": topics,
            "action_items": actions,
            "target_audience": audience,
            "urgency_level": urgency,
            "sentiment": sentiment,
        }

    def _detect_intent(self, text):
        scores = defaultdict(float)
        for intent, patterns in self.INTENT_PATTERNS.items():
            for p in patterns:
                if re.search(p, text, re.IGNORECASE):
                    scores[intent] += 1.0 if intent in ["emergency", "warning"] else 0.8
        if not scores:
            return "unknown", 0.45
        best = max(scores, key=scores.get)
        conf = min(0.5 + scores[best] * 0.14, 0.97)
        return best, conf

    def _extract_topics(self, text, location, situation):
        topics = []
        topic_map = {
            "travel": r'\b(?:train|flight|bus|metro|journey|travel|trip|departure|arrival)\b',
            "time":   r'\b(?:time|schedule|delay|late|early|arriving|departing)\b',
            "safety": r'\b(?:safety|security|emergency|caution|warning)\b',
            "service":r'\b(?:service|help|assistance|support|customer)\b',
            "location":r'\b(?:platform|gate|terminal|station|airport|stop)\b',
            "food":   r'\b(?:food|restaurant|cafe|meal|drink|coffee)\b',
            "health": r'\b(?:health|medical|doctor|hospital)\b',
        }
        for t, pat in topic_map.items():
            if re.search(pat, text, re.IGNORECASE):
                topics.append(t)
        # Infer from location
        loc_topics = {
            "Airport Terminal": ["travel","flight"], "Railway Station": ["travel","train"],
            "Hospital": ["health","medical"], "Restaurant/Cafe": ["food","dining"],
        }
        for t in loc_topics.get(location, []):
            if t not in topics:
                topics.append(t)
        return topics[:6]

    def _extract_actions(self, text):
        actions = []
        pats = [
            (r'(?:please|kindly)\s+([^.!?]{3,50})', "{}"),
            (r'(?:proceed\s+to|go\s+to|move\s+to|line\s+up)\s+([^.!?]{3,50})', "Go to {}"),
            (r'(?:wait\s+(?:at|for|until))\s+([^.!?]{3,50})', "Wait {}"),
            (r'(?:board|enter|exit)\s+([^.!?]{3,50})', "Board/Enter {}"),
        ]
        for pat, tmpl in pats:
            for m in re.findall(pat, text, re.IGNORECASE):
                a = tmpl.format(m.strip()[:45])
                if a not in actions:
                    actions.append(a)
        return actions[:4]

    def _detect_audience(self, text):
        for aud, pats in [
            ("passengers",  [r"passengers", r"travelers", r"commuters"]),
            ("customers",   [r"customers", r"shoppers", r"guests"]),
            ("students",    [r"students", r"learners"]),
            ("patients",    [r"patients", r"visitors"]),
            ("general_public", [r"everyone", r"all", r"ladies", r"gentlemen"]),
        ]:
            for p in pats:
                if re.search(p, text, re.IGNORECASE):
                    return aud
        return "general_public"

    def _assess_urgency(self, text, intent):
        if intent in ["emergency", "warning"]:
            return "critical"
        for level, pats in [
            ("critical", [r"emergency", r"immediately", r"now", r"urgent"]),
            ("high",     [r"quickly", r"hurry", r"right\s+away", r"asap"]),
            ("medium",   [r"soon", r"shortly", r"please"]),
        ]:
            for p in pats:
                if re.search(p, text, re.IGNORECASE):
                    return level
        return "normal"

    def _sentiment(self, text):
        pos = sum(1 for w in ["welcome","thank","please","happy","good","great","wonderful","glad"] if w in text)
        neg = sum(1 for w in ["sorry","unfortunately","delay","cancel","problem","regret","inconvenience"] if w in text)
        if pos > neg + 1: return "positive"
        if neg > pos + 1: return "negative"
        return "neutral"

    def _summary(self, text, intent, entities, audience):
        verbs = {
            "announcement": "announces", "instruction": "instructs", "warning": "warns",
            "question": "asks", "request": "requests", "information": "informs",
            "greeting": "greets", "emergency": "alerts about emergency for",
        }
        verb = verbs.get(intent, "communicates to")
        parts = [f"Speaker {verb} {audience}"]
        nums = entities.get("numbers", [])
        times = entities.get("times", [])
        locs = entities.get("locations", [])
        if nums: parts.append(f"regarding {', '.join(nums[:2])}")
        if times: parts.append(f"at {times[0]}")
        if locs: parts.append(f"at {locs[0]}")
        return ". ".join(parts) + "."

    def _empty(self):
        return {"intent": "unknown", "intent_confidence": 0, "summary": "No text to analyze.",
                "entities": {}, "topics": [], "action_items": [], "target_audience": "unknown",
                "urgency_level": "normal", "sentiment": "neutral"}


# ==============================
# 🧠 CORE ANALYZER  (location + situation + evidence)
# ==============================
class Analyzer:
    def __init__(self, learning):
        self.learning = learning
        self._init_keywords()

    def _init_keywords(self):
        self.loc_kw = {
            "Airport Terminal": {
                "airport": 5, "terminal": 5, "flight": 5, "airline": 5, "aircraft": 5,
                "boarding": 4.5, "gate": 4, "runway": 5, "takeoff": 5, "landing": 5,
                "captain": 4, "pilot": 4.5, "cabin": 4, "heathrow": 5, "lufthansa": 5,
                "emirates": 5, "british airways": 5, "air india": 5, "indigo": 4.5,
                "economy": 3.5, "business class": 4, "passport": 3.5, "baggage": 3.5,
                "departure": 4, "arrival": 4, "passengers": 2.5,
                # International terms
                "аэропорт": 5, "самолет": 5, "рейс": 5, "посадка": 4.5,  # Russian
                "飞机场": 5, "航班": 5,  # Chinese
                "हवाई अड्डा": 5, "विमान": 5,  # Hindi
            },
            "Railway Station": {
                "railway": 5, "train": 5, "platform": 5, "station": 3.5, "locomotive": 5,
                "coach": 4, "compartment": 4, "bogey": 4.5, "rail": 4.5, "track": 3.5,
                "rajdhani": 5, "shatabdi": 5, "indian railways": 5, "irctc": 5,
                "express": 3, "local train": 4, "reservation": 3.5, "engine": 3,
                # International terms
                "вокзал": 5, "поезд": 5, "платформа": 5,  # Russian
                "火车站": 5, "列车": 5,  # Chinese
                "रेलवे स्टेशन": 5, "ट्रेन": 5,  # Hindi
            },
            "Hospital": {
                "hospital": 5, "doctor": 5, "nurse": 5, "patient": 5, "medical": 4.5,
                "emergency": 4, "ambulance": 5, "surgery": 5, "ward": 4.5, "icu": 5,
                "clinic": 4, "medicine": 3.5, "treatment": 4,
                # International terms
                "больница": 5, "врач": 5, "медицина": 4.5,  # Russian
                "医院": 5, "医生": 5,  # Chinese
            },
            "Shopping Mall": {
                "mall": 5, "shopping": 5, "shop": 4, "store": 4, "sale": 3.5,
                "discount": 3.5, "customer": 2.5, "escalator": 4.5, "food court": 5,
                # International terms
                "торговый центр": 5, "магазин": 5,  # Russian
                "购物中心": 5, "商店": 5,  # Chinese
            },
            "Street/Road": {
                "traffic": 5, "road": 4.5, "highway": 5, "street": 4, "car": 3.5,
                "vehicle": 3.5, "signal": 4, "crossing": 3.5, "jam": 4, "horn": 3.5,
                # International terms
                "дорога": 4.5, "улица": 4, "трафик": 5,  # Russian
                "道路": 4.5, "街道": 4,  # Chinese
            },
            "Office Building": {
                "office": 5, "meeting": 4.5, "conference": 5, "presentation": 4.5,
                "manager": 4, "corporate": 4.5, "company": 3.5, "project": 3.5,
                "work": 3, "desk": 3.5, "colleague": 4,
                # International terms
                "офис": 5, "работа": 3.5, "совещание": 4.5,  # Russian
                "办公室": 5, "会议": 4.5,  # Chinese
            },
            "Stadium/Arena": {
                "stadium": 5, "arena": 5, "match": 5, "game": 4, "team": 3.5,
                "score": 4.5, "goal": 5, "cricket": 5, "football": 5, "ipl": 5,
            },
            "Religious Place": {
                "temple": 5, "church": 5, "mosque": 5, "prayer": 4.5, "worship": 4.5,
                "aarti": 5, "namaz": 5, "mandir": 5, "masjid": 5,
                # International terms
                "церковь": 5, "молитва": 4.5,  # Russian
                "寺庙": 5, "教堂": 5,  # Chinese
            },
            "Restaurant/Cafe": {
                "restaurant": 5, "cafe": 5, "food": 3.5, "menu": 4.5, "waiter": 4.5,
                "chef": 5, "table": 3, "order": 3.5, "dinner": 3.5, "lunch": 3.5,
                # International terms
                "ресторан": 5, "кафе": 5, "еда": 3.5,  # Russian
                "餐厅": 5, "咖啡馆": 5,  # Chinese
            },
            "Park/Outdoor": {
                "park": 5, "garden": 5, "outdoor": 4, "nature": 4.5, "tree": 3.5,
                "bird": 3.5, "walk": 2.5, "bench": 4,
            },
            "Metro/Subway": {
                "metro": 5, "subway": 5, "underground": 5, "delhi metro": 5,
                "mumbai metro": 5, "token": 3.5,
                # International terms
                "метро": 5,  # Russian
                "地铁": 5,  # Chinese
            },
            "Construction Site": {
                "construction": 5, "cement": 4.5, "crane": 5, "scaffold": 5,
                "building site": 5, "labor": 3.5,
            },
            "School/University": {
                "school": 5, "university": 5, "college": 5, "class": 4, "student": 4.5,
                "teacher": 4.5, "professor": 5, "lecture": 5, "exam": 4.5,
                # International terms
                "школа": 5, "университет": 5, "студент": 4.5,  # Russian
                "学校": 5, "大学": 5,  # Chinese
            },
        }

        self.sit_kw = {
            "Emergency": {
                "emergency": 5, "help": 4.5, "fire": 5, "accident": 5, "danger": 5,
                "police": 4.5, "ambulance": 5, "injured": 5, "attack": 5, "urgent": 4.5,
            },
            "Boarding/Departure": {
                "boarding": 5, "now boarding": 5, "final call": 5, "departure": 4.5,
                "proceed to": 4.5, "gate": 3.5, "platform": 3.5, "all passengers": 4,
                "economy customers": 5, "continue boarding": 5, "line up": 4.5,
                "wait for departure": 5,
            },
            "Flight Delay": {
                "delayed": 5, "delay": 5, "postponed": 4.5, "rescheduled": 4.5,
                "inconvenience": 4, "regret": 4, "technical": 4,
            },
            "Announcement": {
                "attention": 4.5, "announcement": 5, "ladies and gentlemen": 5,
                "attention please": 5, "information": 4, "kindly": 3.5, "passengers": 3,
            },
            "Traffic": {
                "traffic": 5, "jam": 4.5, "congestion": 5, "heavy traffic": 5,
                "stuck": 4, "slow": 3.5,
            },
            "Busy/Crowded": {
                "crowded": 5, "busy": 4, "rush": 4, "queue": 4, "packed": 4.5,
                "rush hour": 5,
            },
            "Sports Event": {
                "goal": 5, "score": 4.5, "match": 4.5, "win": 4, "tournament": 4.5,
            },
            "Normal/Quiet": {},
        }

    def analyze(self, trans, sounds):
        text = trans.get("text", "").lower()
        raw = trans.get("raw", text).lower()
        reliable = trans.get("is_reliable", False)
        detected = sounds.get("sounds", {})
        loc_hints = sounds.get("location_hints", [])
        sit_hints = sounds.get("situation_hints", [])
        full_text = f"{text} {raw}".lower()
        sound_list = list(detected.keys())

        tw = 0.70 if reliable and text else 0.25
        sw = 1.0 - tw

        loc, loc_conf, loc_ev = self._detect_loc(full_text, loc_hints, tw, sw, sound_list)
        sit, sit_conf, is_emg, sit_ev = self._detect_sit(full_text, sit_hints, tw, sw, sound_list)
        emg_prob = 0.98 if is_emg else self._calc_emg(full_text, sit_hints)
        overall = (loc_conf + sit_conf) / 2
        evidence = self._build_ev(text, reliable, loc_ev, sit_ev, detected, loc_hints)

        return {
            "location": loc, "location_confidence": round(loc_conf, 3),
            "situation": sit, "situation_confidence": round(sit_conf, 3),
            "is_emergency": is_emg, "emergency_probability": round(emg_prob, 3),
            "overall_confidence": round(overall, 3),
            "evidence": evidence, "detected_sounds": sound_list[:10],
            "text_reliable": reliable,
        }

    def _detect_loc(self, text, hints, tw, sw, sounds):
        scores, evidence = {}, {}
        for loc, kws in self.loc_kw.items():
            score, ev = 0.0, []
            kw_score = sum(w for k, w in kws.items() if k in text)
            if kw_score > 0:
                matched = [k for k in kws if k in text][:4]
                score += min(kw_score / 10, 1.5) * tw
                if matched: ev.append(f"Keywords: {', '.join(matched)}")
            for hl, hs in hints:
                if hl == loc and hs > 0.05:
                    score += hs * sw * 2
                    ev.append(f"Sound: {loc}")
            boost = self.learning.get_boost('location', loc, text, sounds)
            if boost > 0: score += boost
            if score > 0.1:
                scores[loc] = score
                evidence[loc] = ev
        if not scores:
            if hints:
                best = hints[0]
                return best[0], 0.55 + best[1]*0.3, [f"Sound suggests: {best[0]}"]
            return "Unknown", 0.40, ["No clear indicators"]
        best = max(scores, key=scores.get)
        conf = 0.70 + min(scores[best] / 2, 0.28)
        return best, round(conf, 3), evidence.get(best, [])

    def _detect_sit(self, text, hints, tw, sw, sounds):
        candidates = []
        for sit, kws in self.sit_kw.items():
            score, ev = 0.0, []
            kw_score = sum(w for k, w in kws.items() if k in text)
            if kw_score > 0:
                matched = [k for k in kws if k in text][:3]
                score += min(kw_score / 8, 1.5) * tw
                if matched: ev.append(f"Keywords: {', '.join(matched)}")
            for hs_sit, hs_score in hints:
                if hs_sit == sit and hs_score > 0.05:
                    score += hs_score * sw * 2
                    ev.append(f"Sound: {sit}")
            boost = self.learning.get_boost('situation', sit, text, sounds)
            score += boost
            is_emg = sit in ["Emergency", "Medical Emergency", "Accident"]
            priority = 10 if is_emg else (7 if sit == "Boarding/Departure" else 5)
            if score > 0.1:
                candidates.append({"sit": sit, "score": score, "priority": priority, "is_emg": is_emg, "ev": ev})
        if not candidates:
            if hints:
                best = hints[0]
                return best[0], 0.60, False, [f"Sound: {best[0]}"]
            return "Normal/Quiet", 0.55, False, ["No specific situation"]
        candidates.sort(key=lambda x: (x["priority"], x["score"]), reverse=True)
        best = candidates[0]
        conf = 0.68 + min(best["score"] / 1.5, 0.28)
        return best["sit"], round(conf, 3), best["is_emg"], best["ev"]

    def _calc_emg(self, text, hints):
        prob = 0.03
        for w, b in {"help": 0.15, "emergency": 0.2, "fire": 0.2, "accident": 0.18}.items():
            if w in text: prob += b
        for sit, score in hints:
            if sit in ["Emergency", "Medical Emergency"]: prob += score * 0.4
        return min(prob, 0.95)

    def _build_ev(self, text, reliable, loc_ev, sit_ev, sounds, hints):
        ev = []
        if reliable and text:
            preview = text[:80] + ("..." if len(text) > 80 else "")
            ev.append(f'Speech: "{preview}"')
        else:
            ev.append("Speech unclear — using sound analysis")
        for e in loc_ev[:2]: ev.append(f"Location → {e}")
        for e in sit_ev[:2]: ev.append(f"Situation → {e}")
        if sounds:
            ev.append(f"Sounds: {', '.join(list(sounds.keys())[:5])}")
        if hints:
            ev.append(f"Hints: {', '.join([f'{h[0]} ({h[1]*100:.0f}%)' for h in hints[:2]])}")
        return ev[:7]


# ==============================
# 📊 PYDANTIC MODELS  (response schemas)
# ==============================
class EmotionData(BaseModel):
    primary_emotion: str
    intensity: float
    micro_emotions: Dict[str, float]
    all_emotions: Dict[str, float]
    valence: float
    arousal: float
    emotional_intensity: float

class SpeakerConfidence(BaseModel):
    score: float
    level: str
    breakdown: Dict[str, float]
    confidence_words_found: List[str]
    hesitation_words_found: List[str]

class QualityData(BaseModel):
    score: float
    max_score: int
    feedback: List[str]

class SemanticData(BaseModel):
    intent: str
    intent_confidence: float
    summary: str
    entities: Dict[str, Any]
    topics: List[str]
    action_items: List[str]
    target_audience: str
    urgency_level: str
    sentiment: str

class LanguageData(BaseModel):
    code: str
    name: str
    native: str
    confidence: float

class AnalysisResult(BaseModel):
    # Core
    location: str
    location_confidence: float
    situation: str
    situation_confidence: float
    is_emergency: bool
    emergency_probability: float
    overall_confidence: float
    # Rich fields
    emotions: EmotionData
    speaker_confidence: SpeakerConfidence
    quality: QualityData
    semantic_analysis: SemanticData
    language: LanguageData
    # Text
    transcribed_text: str
    native_transcribed_text: str
    text_reliable: bool
    # Sounds & Evidence
    detected_sounds: List[str]
    evidence: List[str]
    # Meta
    processing_time_ms: float
    request_id: str
    timestamp: str
    audio_duration: float
    analysis_mode: str

class HealthResponse(BaseModel):
    status: str
    version: str
    models: Dict[str, bool]
    upgraded_mode: bool
    timestamp: str

class FeedbackRequest(BaseModel):
    request_id: str
    correct_location: Optional[str] = None
    correct_situation: Optional[str] = None


# ==============================
# 🌐 GLOBALS
# ==============================
loader = AudioLoader()
whisper = WhisperManager()
yamnet = YAMNetManager()
learning = LearningSystem()
analyzer = Analyzer(learning)
emotion_detector = EmotionDetector()
confidence_scorer = ConfidenceScorer()
quality_scorer = QualityScorer()
entity_extractor = EntityExtractor()
semantic_analyzer = SemanticAnalyzer()
language_detector = LanguageDetector()
history = {}


# ==============================
# 🚀 LIFESPAN
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("🚀 AURALIS v5.0 — UPGRADED STANDALONE")
    print("="*60)
    setup_ffmpeg()
    whisper.load()
    yamnet.load()
    print("\n" + "="*60)
    print("✅ READY")
    print("="*60)
    print(f"🌐 http://127.0.0.1:8000")
    print(f"📚 http://127.0.0.1:8000/docs")
    print(f"📍 {len(LOCATIONS)} locations | 🎯 {len(SITUATIONS)} situations")
    print(f"🔊 {len(yamnet.labels)} sounds | 📚 {learning.corrections} corrections")
    print(f"😊 Emotions | 🎤 Speaker Confidence | 📊 Quality | 🧠 Semantic")
    print("="*60 + "\n")
    yield
    learning.save()
    print("💾 Saved | 👋 Bye")


# ==============================
# 🚀 APP
# ==============================
app = FastAPI(title="Auralis", version="5.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ==============================
# 📍 ROUTES
# ==============================
@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy", version="5.0.0",
        models={"whisper": whisper.loaded, "yamnet": yamnet.loaded},
        upgraded_mode=True,
        timestamp=datetime.now().isoformat()
    )

@app.get("/labels")
async def labels():
    return {
        "locations": LOCATIONS, "situations": SITUATIONS,
        "num_locations": len(LOCATIONS), "num_situations": len(SITUATIONS),
        "sounds": len(yamnet.labels)
    }


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    start = time.time()
    rid = uuid.uuid4().hex[:8]
    tmp = tempfile.gettempdir()
    ext = os.path.splitext(file.filename)[1] or ".wav"
    orig = os.path.join(tmp, f"aur_{rid}{ext}")
    wav  = os.path.join(tmp, f"aur_{rid}.wav")

    print("\n" + "="*60)
    print(f"📥 {file.filename} | ID: {rid}")
    print("="*60)

    try:
        content = await file.read()
        with open(orig, "wb") as f:
            f.write(content)
        print(f"💾 {len(content)/1024:.1f} KB")

        # ── Load audio ──
        print("🔊 Loading...")
        audio, dur = loader.load(orig)
        print(f"⏱️{dur:.2f}s")
        if dur < MIN_DURATION:
            raise HTTPException(400, "Audio too short (< 0.5s)")
        if dur > MAX_DURATION:
            audio = audio[:int(SAMPLE_RATE * MAX_DURATION)]
            dur = MAX_DURATION
        loader.save_wav(audio, wav)

        # ── Transcribe (English + native) ──
        print("🎤 Transcribing...")
        trans = whisper.transcribe(wav)
        if trans.get("is_reliable"):
            print(f"📝 ✓ '{trans['text'][:70]}...'")
        else:
            print("📝 ⚠️Unreliable")

        # ── Detect language from native text ──
        native_text = trans.get("native_text", "")
        lang_info = language_detector.detect(native_text if native_text else trans.get("text", ""))
        trans["detected_language"] = lang_info["code"]
        print(f"🌍 Language: {lang_info['name']} ({lang_info['code']})")

        # ── Sound analysis ──
        print("🔊 Sounds...")
        snd = yamnet.analyze(audio)
        if snd["sounds"]:
            print(f"🔊 {list(snd['sounds'].keys())[:5]}")
        if snd["location_hints"]:
            print(f"💡 {[h[0] for h in snd['location_hints'][:3]]}")

        # ── Core location / situation ──
        print("🧠 Analyzing...")
        res = analyzer.analyze(trans, snd)

        # ── Emotions ──
        emo_result = emotion_detector.analyze(audio, trans.get("text", ""), snd.get("sounds", {}))

        # ── Speaker Confidence ──
        conf_result = confidence_scorer.score(audio, trans.get("text", ""), trans.get("confidence", 0.5))

        # ── Quality Score ──
        qual_result = quality_scorer.score(trans, snd, res["location_confidence"], res["situation_confidence"])

        # ── Semantic Analysis ──
        sem_result = semantic_analyzer.analyze(
            trans.get("text", ""),
            res["location"], res["situation"],
            res["detected_sounds"]
        )

        # ── Store history ──
        history[rid] = {"trans": trans, "sounds": snd, "result": res}
        if len(history) > 500:
            del history[list(history.keys())[0]]

        proc_ms = round((time.time() - start) * 1000, 1)

        # ── Build response ──
        response = AnalysisResult(
            # Core
            location=res["location"],
            location_confidence=res["location_confidence"],
            situation=res["situation"],
            situation_confidence=res["situation_confidence"],
            is_emergency=res["is_emergency"],
            emergency_probability=res["emergency_probability"],
            overall_confidence=res["overall_confidence"],
            # Rich
            emotions=EmotionData(**emo_result),
            speaker_confidence=SpeakerConfidence(**conf_result),
            quality=QualityData(**qual_result),
            semantic_analysis=SemanticData(**sem_result),
            language=LanguageData(**lang_info),
            # Text
            transcribed_text=trans.get("text", ""),
            native_transcribed_text=native_text,
            text_reliable=trans.get("is_reliable", False),
            # Sounds & Evidence
            detected_sounds=res["detected_sounds"],
            evidence=res["evidence"],
            # Meta
            processing_time_ms=proc_ms,
            request_id=rid,
            timestamp=datetime.now().isoformat(),
            audio_duration=round(dur, 2),
            analysis_mode="upgraded_v5",
        )

        # ── Console summary ──
        print(f"\n✅ RESULT:")
        print(f"   📍 {response.location} ({response.location_confidence*100:.0f}%)")
        print(f"   🎯 {response.situation} ({response.situation_confidence*100:.0f}%)")
        print(f"   😊 Emotion: {emo_result['primary_emotion']} (intensity {emo_result['intensity']})")
        print(f"   🎤 Speaker confidence: {conf_result['score']}/10 ({conf_result['level']})")
        print(f"   📊 Quality: {qual_result['score']}/10")
        print(f"   🚨 Emergency: {response.is_emergency}")
        print(f"   🌍 Language: {lang_info['name']}")
        print(f"   ⏱️{proc_ms:.0f}ms")
        print("="*60 + "\n")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        for f in [orig, wav]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.request_id not in history:
        return {"status": "not_found"}
    h = history[req.request_id]
    if req.correct_location:
        learning.learn('location', req.correct_location, h['trans'].get('text', ''),
                      list(h['sounds'].get('sounds', {}).keys()))
    if req.correct_situation:
        learning.learn('situation', req.correct_situation, h['trans'].get('text', ''),
                      list(h['sounds'].get('sounds', {}).keys()))
    return {"status": "learned", "corrections": learning.corrections}


@app.get("/stats")
async def stats():
    return {
        "version": "5.0.0", "analyses": len(history),
        "models": {"whisper": whisper.loaded, "yamnet": yamnet.loaded},
        "corrections": learning.corrections,
        "features": ["emotions", "speaker_confidence", "quality_score",
                     "semantic_analysis", "entity_extraction", "language_detection"]
    }


# Auth mocks (kept for compatibility)
@app.post("/login")
async def login(d: dict): return {"token": uuid.uuid4().hex[:16]}
@app.post("/register")
async def register(d: dict): return {"status": "ok"}
@app.post("/save_history")
async def save_hist(d: dict): return {"id": uuid.uuid4().hex[:8]}


# ==============================
# 🏃 RUN
# ==============================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Auralis v5.0...\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")