import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio
import librosa
import librosa.display
from pathlib import Path
from typing import Dict
from .config import STEM_COLORS, FIGURE_SIZE, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class PlottingError(Exception):
    """Custom exception for plotting errors."""
    pass

def get_common_audio_name(stem_dict: Dict[str, str]) -> str:
    """Extract the common audio name from stem filenames.

    Args:
        stem_dict: Dictionary mapping stem names to file paths.

    Returns:
        The common audio name prefix (without stem suffixes).

    Raises:
        ValueError: If no common prefix is found.
    """
    stem_paths = [Path(path).stem for path in stem_dict.values()]
    if not stem_paths:
        raise ValueError("Empty stem dictionary provided")
    
    # Find the longest common prefix
    def common_prefix(strings):
        if not strings:
            return ""
        shortest = min(strings, key=len)
        for i, char in enumerate(shortest):
            for other in strings:
                if other[i] != char:
                    return shortest[:i]
        return shortest
    
    # Remove known stem suffixes to get the base name
    stem_names = ['drums', 'bass', 'other', 'vocals', 'piano', 'guitar']
    cleaned_names = []
    for path in stem_paths:
        for stem in stem_names:
            if path.lower().endswith(f'_{stem}'):
                cleaned_names.append(path[:-len(f'_{stem}')])
                break
        else:
            cleaned_names.append(path)
    
    common_name = common_prefix(cleaned_names)
    if not common_name:
        raise ValueError("Could not determine common audio name from stem files")
    return common_name

class AudioPlotter:
    def __init__(self):
        """Initialize the AudioPlotter."""
        self.sample_rate = DEFAULT_CONFIG['sample_rate']
        # Set seaborn theme with white background for vivid aesthetics
        sns.set_theme(style="darkgrid", palette="husl", font_scale=1.3, rc={"grid.alpha": 0.7, "axes.facecolor": "#FFFFFF"})

    def plot_waveforms(self, stem_dict: Dict[str, str], output_dir: str) -> None:
        """Generate a single waveform plot with subplots for all stems in n rows, 1 column.

        Args:
            stem_dict: Dictionary mapping stem names to file paths.
            output_dir: Directory to save the plot.

        Raises:
            PlottingError: If waveform plotting fails.
        """
        try:
            # Create a single figure with n subplots (n rows, 1 column)
            num_stems = len(stem_dict)
            fig, axes = plt.subplots(num_stems, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * num_stems / 4))
            if num_stems == 1:
                axes = [axes]  # Ensure axes is iterable for single stem
            audio_name = get_common_audio_name(stem_dict)

            for idx, (stem_name, stem_path) in enumerate(stem_dict.items()):
                waveform, _ = torchaudio.load(stem_path)
                # Use left channel for plotting
                waveform = waveform[0].numpy()

                # Downsample waveform to ~1000 points for visualization
                num_points = 1000
                if len(waveform) > num_points:
                    indices = np.linspace(0, len(waveform) - 1, num_points, dtype=int)
                    waveform = waveform[indices]

                # Generate time axis (in seconds)
                time_axis = np.linspace(0, len(waveform) / self.sample_rate, len(waveform))

                # Plot waveform with vivid styling
                axes[idx].plot(time_axis, waveform, color=STEM_COLORS.get(stem_name, '#000000'), linewidth=2)
                axes[idx].set_title(f"{stem_name.capitalize()}", fontsize=16, pad=12, weight='bold')
                axes[idx].set_xlabel("Time (seconds)", fontsize=14)
                axes[idx].set_ylabel("Amplitude", fontsize=14)
                axes[idx].grid(True, linestyle='--', alpha=0.7)

            # Adjust layout manually to avoid tight_layout issues
            fig.subplots_adjust(left=0.1, right=0.8, top=0.95, bottom=0.05, hspace=0.4)
            plt.suptitle(f"Waveforms for {audio_name}", fontsize=18, y=1.02, weight='bold')

            # Save combined waveform plot
            plot_path = os.path.join(output_dir, f"{audio_name}_combined_waveforms.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=400)
            plt.close()
            logger.info(f"Combined waveform plot saved at {plot_path}")
        except Exception as e:
            logger.error(f"Error plotting waveforms: {str(e)}")
            raise PlottingError(f"Error plotting waveforms: {str(e)}")

    def plot_spectrograms(self, stem_dict: Dict[str, str], output_dir: str) -> None:
        """Generate a single spectrogram plot with subplots for all stems in n rows, 1 column.

        Args:
            stem_dict: Dictionary mapping stem names to file paths.
            output_dir: Directory to save the plot.

        Raises:
            PlottingError: If spectrogram plotting fails.
        """
        try:
            # Create a single figure with n subplots (n rows, 1 column)
            num_stems = len(stem_dict)
            fig, axes = plt.subplots(num_stems, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * num_stems / 4))
            if num_stems == 1:
                axes = [axes]  # Ensure axes is iterable for single stem
            audio_name = get_common_audio_name(stem_dict)

            # Store the image object for the colorbar
            last_im = None
            for idx, (stem_name, stem_path) in enumerate(stem_dict.items()):
                waveform, sr = torchaudio.load(stem_path)
                # Use left channel and convert to numpy
                waveform = waveform[0].numpy()

                # Compute spectrogram using librosa
                D = librosa.stft(waveform, n_fft=2048, hop_length=512)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

                # Plot spectrogram with frequency on x-axis and time on y-axis
                librosa.display.specshow(
                    S_db,
                    sr=self.sample_rate,
                    hop_length=512,
                    x_axis='hz',  # Frequency on x-axis
                    y_axis='time',  # Time on y-axis
                    ax=axes[idx],
                    cmap='magma'  # Vibrant colormap
                )
                axes[idx].set_title(f"{stem_name.capitalize()}", fontsize=16, pad=12, weight='bold')
                axes[idx].set_xlabel("Frequency (Hz)", fontsize=14)
                axes[idx].set_ylabel("Time (seconds)", fontsize=14)
                last_im = axes[idx].collections[-1]  # Store the QuadMesh for colorbar

            # Add a shared colorbar outside the plot with increased padding
            cbar = fig.colorbar(last_im, ax=axes, location='right', pad=0.1, shrink=0.8)
            cbar.set_label("Amplitude (dB)", fontsize=14, weight='bold')
            cbar.ax.tick_params(labelsize=12)

            # Adjust layout manually to ensure colorbar is outside and avoid tight_layout issues
            fig.subplots_adjust(left=0.1, right=0.75, top=0.95, bottom=0.05, hspace=0.4)
            plt.suptitle(f"Spectrograms for {audio_name}", fontsize=18, y=1.02, weight='bold')

            # Save combined spectrogram plot
            plot_path = os.path.join(output_dir, f"{audio_name}_combined_spectrograms.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=400)
            plt.close()
            logger.info(f"Combined spectrogram plot saved at {plot_path}")
        except Exception as e:
            logger.error(f"Error plotting spectrograms: {str(e)}")
            raise PlottingError(f"Error plotting spectrograms: {str(e)}")
