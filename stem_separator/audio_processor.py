import os
import logging
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Any
from .config import SUPPORTED_FORMATS, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass

class StemSeparator:
    def __init__(self, model: Any, model_type: str):
        """Initialize the StemSeparator.

        Args:
            model: The initialized model (torchaudio or audio-separator).
            model_type: The model type ('torchaudio' or 'audio-separator').
        """
        self.model = model
        self.model_type = model_type
        self.sample_rate = DEFAULT_CONFIG['sample_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if model_type == 'torchaudio' else 'cpu'

    def _extract_waveform(self, audio_path: str) -> torch.Tensor:
        """Extract and preprocess waveform from audio file (for torchaudio).

        Args:
            audio_path: Path to the audio file.

        Returns:
            The preprocessed waveform tensor.

        Raises:
            AudioProcessingError: If audio loading or processing fails.
        """
        try:
            waveform, original_sample_rate = torchaudio.load(audio_path)
            logger.info(f"{audio_path} - Loaded with sample rate {original_sample_rate} Hz")

            # Ensure stereo
            #if waveform.shape[0] == 1:
            #    waveform = waveform.repeat(2, 1)
            #    logger.info("Converted mono to stereo")

            # Resample to 44100 Hz if necessary
            if original_sample_rate != self.sample_rate:
                logger.info(f"Resampling from {original_sample_rate} Hz to {self.sample_rate} Hz")
                resampler = torchaudio.transforms.Resample(original_sample_rate, self.sample_rate)
                waveform = resampler(waveform)

            return waveform
        except RuntimeError as e:
            logger.error(f"Failed to load audio {audio_path}: {str(e)}")
            logger.error("Ensure 'sox_io' backend is available. Install libsox: e.g., 'apt-get install sox' or 'pip install soundfile'.")
            raise AudioProcessingError(f"Failed to load audio {audio_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to process audio {audio_path}: {str(e)}")
            raise AudioProcessingError(f"Failed to process audio {audio_path}: {str(e)}")

    def separate_stems(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Separate audio stems and save them to output directory as WAV.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save separated stems.

        Returns:
            A dictionary mapping stem names to file paths.

        Raises:
            AudioProcessingError: If stem separation fails.
        """
        try:
            if not os.path.exists(audio_path):
                raise AudioProcessingError(f"Audio file not found: {audio_path}")

            if self.model_type == 'torchaudio':
                waveform = self._extract_waveform(audio_path)
                stem_dict = self._process_torchaudio_stems(waveform, audio_path, output_dir)
            else:  # audio-separator
                stem_dict = self._process_audio_separator_stems(audio_path, output_dir)
            return stem_dict
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            raise AudioProcessingError(f"Error processing {audio_path}: {str(e)}")

    def _process_torchaudio_stems(self, waveform: torch.Tensor, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Process stems using the torchaudio model and save as WAV.

        Args:
            waveform: Input waveform tensor.
            audio_path: Path to the input audio file.
            output_dir: Directory to save stems.

        Returns:
            A dictionary mapping stem names to file paths.

        Raises:
            AudioProcessingError: If stem processing or saving fails.
        """
        stem_dict = {}
        try:
            logger.info("Separating stems with torchaudio...")
            with torch.no_grad():
                stems = self.model(waveform.unsqueeze(0).to(self.device))
            logger.info("Stem separation completed")

            os.makedirs(output_dir, exist_ok=True)
            for idx, name in enumerate(self.model.sources):
                stem_path = os.path.join(output_dir, f"{Path(audio_path).stem}_{name}.wav")
                try:
                    torchaudio.save(stem_path, stems[0, idx].cpu(), self.sample_rate, format='wav')
                    stem_dict[name] = stem_path
                    logger.info(f"Stem {name} saved at {stem_path}")
                except RuntimeError as e:
                    logger.error(f"Failed to save stem {name} as WAV: {str(e)}")
                    raise AudioProcessingError(f"Failed to save stem {name} as WAV: {str(e)}")
            return stem_dict
        except Exception as e:
            logger.error(f"Error processing stems for {audio_path}: {str(e)}")
            raise AudioProcessingError(f"Error processing stems for {audio_path}: {str(e)}")

    def _process_audio_separator_stems(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Process stems using the audio-separator model and save as WAV.

        Args:
            audio_path: Path to the input audio file.
            output_dir: Directory to save stems.

        Returns:
            A dictionary mapping stem names to file paths.

        Raises:
            AudioProcessingError: If stem processing or saving fails.
        """
        stem_dict = {}
        try:
            logger.info("Separating stems with audio-separator...")
            stem_paths = self.model.separate(audio_path)
            logger.info("Stem separation completed")

            os.makedirs(output_dir, exist_ok=True)
            for path in stem_paths:
                full_path = os.path.join(output_dir, path)
                filename = os.path.basename(path).lower()
                if 'vocal' in filename:
                    stem_name = 'vocals'
                elif 'drum' in filename:
                    stem_name = 'drums'
                elif 'bass' in filename:
                    stem_name = 'bass'
                elif 'other' in filename:
                    stem_name = 'other'
                elif 'piano' in filename:
                    stem_name = 'piano'
                elif 'guitar' in filename:
                    stem_name = 'guitar'
                else:
                    logger.warning(f"Unknown stem in filename: {filename}, skipping")
                    continue
                stem_dict[stem_name] = full_path
                logger.info(f"Stem {stem_name} saved at {full_path}")
            return stem_dict
        except Exception as e:
            logger.error(f"Error processing stems for {audio_path}: {str(e)}")
            raise AudioProcessingError(f"Error processing stems for {audio_path}: {str(e)}")
