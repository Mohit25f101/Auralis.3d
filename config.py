# ==============================
# 📄 config.py
# ==============================
# Central configuration for Auralis ML System
# ==============================

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ==============================
# 📁 PATH CONFIGURATION
# ==============================

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Directory paths
WEIGHTS_DIR = BASE_DIR / "weights"
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"
AUDIO_DIR = DATASET_DIR / "audio"
FEATURES_DIR = DATASET_DIR / "features"
METADATA_DIR = DATASET_DIR / "metadata"

# Create directories if they don't exist
for dir_path in [WEIGHTS_DIR, DATASET_DIR, LOGS_DIR, AUDIO_DIR, FEATURES_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    
# Create subdirectories for audio
for split in ['train', 'val', 'test']:
    (AUDIO_DIR / split).mkdir(parents=True, exist_ok=True)

# FFmpeg path (UPDATE THIS FOR YOUR SYSTEM)
FFMPEG_BIN_DIR = r"D:\photo\ffmpeg\ffmpeg-2026-01-07-git-af6a1dd0b2-full_build\bin"


# ==============================
# 🎵 AUDIO CONFIGURATION
# ==============================

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    f_min: int = 80
    f_max: int = 7600
    max_duration: float = 30.0
    min_duration: float = 0.5


# ==============================
# 🏷️ LABEL CONFIGURATION
# ==============================

@dataclass
class LabelConfig:
    """Label definitions for classification"""
    
    locations: List[str] = field(default_factory=lambda: [
        "Airport Terminal",
        "Railway Station",
        "Bus Terminal",
        "Hospital",
        "Shopping Mall",
        "Office",
        "School/University",
        "Restaurant/Cafe",
        "Street/Road",
        "Home/Residential",
        "Park/Outdoor",
        "Stadium/Arena",
        "Unknown"
    ])
    
    situations: List[str] = field(default_factory=lambda: [
        "Normal/Quiet",
        "Busy/Crowded",
        "Emergency",
        "Boarding/Departure",
        "Waiting",
        "Traffic",
        "Meeting/Conference",
        "Announcement",
        "Celebration/Event",
        "Construction",
        "Weather Event",
        "Accident",
        "Medical Emergency",
        "Security Alert",
        "Unknown"
    ])
    
    location_hierarchy: Dict[str, List[int]] = field(default_factory=lambda: {
        "transportation": [0, 1, 2],
        "public_indoor": [3, 4, 5, 6, 7],
        "outdoor": [8, 10, 11],
        "private": [9],
    })
    
    situation_hierarchy: Dict[str, List[int]] = field(default_factory=lambda: {
        "normal": [0, 1, 4],
        "emergency": [2, 11, 12, 13],
        "transit": [3, 5],
        "activity": [6, 7, 8, 9],
        "environmental": [10],
    })
    
    @property
    def num_locations(self) -> int:
        return len(self.locations)
    
    @property
    def num_situations(self) -> int:
        return len(self.situations)


# ==============================
# 🧠 MODEL CONFIGURATION
# ==============================

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    transformer_layers: int = 6
    d_model: int = 256
    num_heads: int = 8
    dff: int = 1024
    dropout_rate: float = 0.1
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    classifier_dropout: float = 0.3
    num_cross_attention_layers: int = 3
    contrastive_projection_dim: int = 128
    contrastive_temperature: float = 0.07


# ==============================
# 🏋️ TRAINING CONFIGURATION
# ==============================

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_learning_rate: float = 1e-7
    patience: int = 10
    min_delta: float = 1e-4
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    augmentation_probability: float = 0.7
    augmentation_intensity: str = "medium"
    contrastive_epochs: int = 50
    contrastive_batch_size: int = 64
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.6
    uncertainty_threshold: float = 0.6
    active_learning_samples: int = 100


# ==============================
# 🌐 API CONFIGURATION
# ==============================

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_file_size: int = 50 * 1024 * 1024
    temp_dir: str = "temp"


# ==============================
# 📦 GLOBAL CONFIG INSTANCES
# ==============================

AUDIO_CONFIG = AudioConfig()
LABEL_CONFIG = LabelConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
API_CONFIG = APIConfig()


# ==============================
# 🛠️ UTILITY FUNCTIONS
# ==============================

def get_device():
    """Get available compute device"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return f"GPU: {gpus[0].name}"
    except:
        pass
    return "CPU"


def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("🔧 AURALIS CONFIGURATION")
    print("="*60)
    print(f"\n📁 Directories:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Weights: {WEIGHTS_DIR}")
    print(f"   Dataset: {DATASET_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    print(f"\n🎵 Audio:")
    print(f"   Sample Rate: {AUDIO_CONFIG.sample_rate} Hz")
    print(f"   Mel Bands: {AUDIO_CONFIG.n_mels}")
    print(f"\n🏷️ Labels:")
    print(f"   Locations: {LABEL_CONFIG.num_locations}")
    print(f"   Situations: {LABEL_CONFIG.num_situations}")
    print(f"\n🧠 Model:")
    print(f"   Transformer Layers: {MODEL_CONFIG.transformer_layers}")
    print(f"   Model Dimension: {MODEL_CONFIG.d_model}")
    print(f"\n🏋️ Training:")
    print(f"   Batch Size: {TRAINING_CONFIG.batch_size}")
    print(f"   Learning Rate: {TRAINING_CONFIG.learning_rate}")
    print(f"\n💻 Device: {get_device()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config()