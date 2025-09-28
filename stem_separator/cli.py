import argparse
import logging
import os
from pathlib import Path
from typing import Any
import torch
import torchaudio
from .config import SUPPORTED_FORMATS, DEFAULT_CONFIG, LOG_FORMAT, LOG_LEVEL
from .models import setup_model, ModelSetupError
from .audio_processor import StemSeparator, AudioProcessingError
from .plotter import AudioPlotter, PlottingError

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

def process_audio_files(args: Any) -> None:
    """Process single audio file or all files in a folder.

    Args:
        args: Parsed command-line arguments.

    Raises:
        AudioProcessingError: If audio processing fails.
        PlottingError: If plotting fails.
    """
    config = {
        'model_type': args.model,
        'download': args.download,
        'model_path': args.model_path,
        'model_filename': args.model_filename,
        'model_file_dir': args.model_file_dir,
        'output_dir': args.output_dir,
        'device': 'cuda' if args.model == 'torchaudio' and torch.cuda.is_available() else 'cpu'
    }

    try:
        model = setup_model(config)
        separator = StemSeparator(model, args.model)
        plotter = AudioPlotter()

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Process single file or directory
        if os.path.isfile(args.input_path):
            if Path(args.input_path).suffix.lower() not in SUPPORTED_FORMATS:
                raise AudioProcessingError(f"Input file {args.input_path} has unsupported format. Choose from {SUPPORTED_FORMATS}")
            stem_dict = separator.separate_stems(args.input_path, args.output_dir)
            plotter.plot_waveforms(stem_dict, args.output_dir)
            plotter.plot_spectrograms(stem_dict, args.output_dir)
        else:
            for file in os.listdir(args.input_path):
                if file.lower().endswith(tuple(SUPPORTED_FORMATS)):
                    audio_path = os.path.join(args.input_path, file)
                    stem_dict = separator.separate_stems(audio_path, args.output_dir)
                    plotter.plot_waveforms(stem_dict, args.output_dir)
                    plotter.plot_spectrograms(stem_dict, args.output_dir)
    except (ModelSetupError, AudioProcessingError, PlottingError) as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

def main():
    """Main function to parse arguments and process audio files."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Separate audio stems using HDemucs (torchaudio or audio-separator) and plot combined waveforms and spectrograms"
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help="Path to audio file or directory containing audio files (.mp3, .wav)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_CONFIG['output_dir'],
        help=f"Directory to save separated stems and plots (default: {DEFAULT_CONFIG['output_dir']})"
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['torchaudio', 'audio-separator'],
        default=DEFAULT_CONFIG['model_type'],
        help=f"Stem separation model to use: torchaudio (4 stems) or audio-separator (4 or 6 stems via --model_filename) (default: {DEFAULT_CONFIG['model_type']})"
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help="Download pretrained model instead of using local model (only for torchaudio)"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_CONFIG['model_path'],
        help="Path to local pretrained model file (required for torchaudio if --download is not set)"
    )
    parser.add_argument(
        '--model_filename',
        type=str,
        default=DEFAULT_CONFIG['model_filename'],
        help=f"Model filename for audio-separator (e.g., htdemucs.yaml for 4 stems, htdemucs_6s.yaml for 6 stems) (default: {DEFAULT_CONFIG['model_filename']})"
    )
    parser.add_argument(
        '--model_file_dir',
        type=str,
        default=DEFAULT_CONFIG['model_file_dir'],
        help=f"Directory containing audio-separator model files (default: {DEFAULT_CONFIG['model_file_dir']})"
    )

    args = parser.parse_args()

    # Validate model path for torchaudio
    if args.model == 'torchaudio' and not args.download and not args.model_path:
        raise ValueError("Model path must be provided for torchaudio when not downloading model")

    # Set up torchaudio backend
    if args.model == 'torchaudio':
        try:
            torchaudio.set_audio_backend("sox_io")
            logger.info("Using sox_io backend for audio processing")
        except RuntimeError as e:
            logger.warning(f"sox_io backend not available: {str(e)}")
            logger.warning("Falling back to default backend. Install 'libsox' or 'soundfile' for full format support.")

    process_audio_files(args)

if __name__ == "__main__":
    main()
