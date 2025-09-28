import logging
import os
import torch
import torchaudio
from audio_separator.separator import Separator
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelSetupError(Exception):
    """Custom exception for model setup errors."""
    pass

def setup_torchaudio_model(config: Dict[str, Any]) -> Any:
    """Set up the torchaudio HDemucs model (4 stems only).

    Args:
        config: Configuration dictionary with 'download', 'model_path', 'device'.

    Returns:
        The initialized torchaudio model.

    Raises:
        ModelSetupError: If model loading fails.
    """
    logger.info("Loading torchaudio model (HDEMUCS_HIGH_MUSDB_PLUS, 4 stems)...")
    try:
        if config['download']:
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model()
        else:
            if not config['model_path'] or not os.path.exists(config['model_path']):
                raise ModelSetupError(f"Model file not found: {config['model_path']}")
            sources = ["drums", "bass", "other", "vocals"]
            model = torchaudio.models.hdemucs_high(sources=sources)
            model_state = torch.load(config['model_path'], map_location=config['device'])
            model.load_state_dict(
                model_state["model"] if "model" in model_state else model_state,
                strict=True
            )
        model.to(config['device'])
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load torchaudio model: {str(e)}")
        raise ModelSetupError(f"Failed to load torchaudio model: {str(e)}")

def setup_audio_separator(config: Dict[str, Any]) -> Separator:
    """Set up the audio-separator model (supports 4 or 6 stems via model_filename).

    Args:
        config: Configuration dictionary with 'model_filename', 'model_file_dir', 'output_dir'.

    Returns:
        The initialized Separator object.

    Raises:
        ModelSetupError: If model loading fails.
    """
    logger.info(f"Loading audio-separator model ({config['model_filename']})...")
    try:
        separator = Separator(
            output_dir=config['output_dir'],
            model_file_dir=config['model_file_dir']
        )
        separator.load_model(model_filename=config['model_filename'])
        return separator
    except Exception as e:
        logger.error(f"Failed to load audio-separator model: {str(e)}")
        raise ModelSetupError(f"Failed to load audio-separator model: {str(e)}")

def setup_model(config: Dict[str, Any]) -> Any:
    """Set up the model based on configuration.

    Args:
        config: Configuration dictionary with 'model_type', 'download', 'model_path',
                'model_filename', 'model_file_dir', 'device', 'output_dir'.

    Returns:
        The initialized model (torchaudio or audio-separator).

    Raises:
        ModelSetupError: If model type is invalid or setup fails.
    """
    if config['model_type'] == 'torchaudio':
        return setup_torchaudio_model(config)
    elif config['model_type'] == 'audio-separator':
        return setup_audio_separator(config)
    else:
        raise ModelSetupError(f"Invalid model type: {config['model_type']}")
