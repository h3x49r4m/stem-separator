import os
from pathlib import Path
from typing import Set

# Supported input audio formats
SUPPORTED_FORMATS: Set[str] = {'.mp3', '.wav'}

# Default configuration
DEFAULT_CONFIG = {
    'sample_rate': 44100,
    'output_dir': 'output_stems',
    'model_type': 'torchaudio',
    'model_filename': 'htdemucs_6s.yaml',
    'model_file_dir': 'models',
    'download': False,
    'model_path': '',
}

# Colors for stem plotting (vivid colors for enhanced visuals)
STEM_COLORS = {
    'drums': '#1e90ff',   # Blue
    'bass': '#ff4500',    # OrangeRed
    'other': '#32cd32',   # LimeGreen
    'vocals': '#ff69b4',  # HotPink
    'piano': '#8a2be2',   # BlueViolet
    'guitar': '#ffa500'   # Orange
}

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Default figure size for plots (base for 4 stems; scales dynamically)
FIGURE_SIZE = (12, 16)
