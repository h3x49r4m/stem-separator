.. Stem Separator
.. =================

Overview
========

Stem Separator is a command-line tool for separating audio files into individual stems (e.g., vocals, drums, bass, other, piano, guitar) using deep learning models from torchaudio (HDemucs) or audio-separator (Demucs variants). It also generates waveform and spectrogram plots for visualization.

- torchaudio (HDemucs): vocals + drums + bass + other
- audio-separator (Demucs variants): vocals + drums + bass + guitar + piano + other

Stem Waveforms

.. raw:: html

    <table align="center">
      <tr>
        <td>audio-separator htdemucs_6s<br><img src="docs/images/only_time_combined_waveforms_audio_separator_htdemucs_6s.png" width="200"></td>
        <td>audio-separator htdemucs_ft<br><img src="docs/images/only_time_combined_waveforms_audio_separator_htdemucs_ft.png" width="200"></td>
        <td>torchaudio hdemucs_high<br><img src="docs/images/only_time_combined_waveforms_torchaudio_hdemucs_high.png" width="200"></td>
      </tr>
    </table>

Features
========

- **Multiple Model Backends:** Choose between a `Torchaudio` model for 4-stem separation or the `audio-separator` library for more flexibility, including 6-stem separation.
- **Batch Processing:** Process a single audio file or an entire directory of audio files.
- **Visualization:** Automatically generates combined waveform and spectrogram plots for the separated stems.
- **Easy to Use:** Simple command-line interface with sensible defaults.

Installation
============

1. **Clone the repository:**

   .. code-block:: bash

      $ git clone https://github.com/h3x49r4m/stem-separator.git
      $ cd stem-separator

2. **Install dependencies:**

   .. code-block:: bash

      $ uv install

   **Note:** You may need to install ``sox`` for ``torchaudio`` to work correctly with all audio formats. On Debian/Ubuntu, you can install it with:

   .. code-block:: bash

      $ sudo apt-get install sox


Model Download:

- torchaudio: HDEMUCS_HIGH_MUSDB_PLUS downloaded from https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt
- audio-separator: For audio-separator, place model YAML files (e.g., htdemucs.yaml, htdemucs_6s.yaml) in the models/ directory. Models reference from https://github.com/facebookresearch/demucs.

Usage
=====

The main script is ``main.py``, which can be run from the command line.

**Using the audio-separator model:**

.. code-block:: bash

   # torchaudio models: download (4 stems)
   $ uv run main.py --input_path <your_song.mp3> --output_dir <output_dir> --model torchaudio --download
   $ uv run main.py --input_path <audio_dir> --output_dir <output_dir> --model torchaudio --download

   # torchaudio models: local models (4 stems)
   $ uv run main.py --input_path <your_song.mp3> --output_dir <output_dir> --model torchaudio --model_path <model_file.pt>
   $ uv run main.py --input_path <audio_dir> --output_dir <output_dir> --model torchaudio --model_path <model_file.pt>


   # audio-separator: htdemucs_ft (4 stems)
   $ uv run main.py --input_path <your_song.mp3> --output_dir <output_dir> --model audio-separator --model_filename htdemucs_ft.yaml --model_file_dir <models_dir>
   $ uv run main.py --input_path <audio_dir> --output_dir <output_dir> --model audio-separator --model_filename htdemucs_ft.yaml --model_file_dir <models_dir>

   # audio-separator: htdemucs_6s (6 stems)
   $ uv run main.py --input_path <your_song.mp3> --output_dir <output_dir> --model audio-separator --model_filename htdemucs_6s.yaml --model_file_dir <models_dir>
   $ uv run main.py --input_path <audio_dir> --output_dir <output_dir> --model audio-separator --model_filename htdemucs_6s.yaml --model_file_dir <models_dir>


Command-Line Arguments
----------------------

- ``--input_path``: (Required) Path to the input audio file or directory.
- ``--output_dir``: (Optional) Directory to save the separated stems and plots. Default: ``output_stems``.
- ``--model``: (Optional) The model to use for separation. Choices: ``torchaudio``, ``audio-separator``. Default: ``torchaudio``.
- ``--download``: (Optional) Download the pre-trained Torchaudio model instead of using a local file.
- ``--model_path``: (Optional) Path to a local pre-trained model file (for Torchaudio).
- ``--model_filename``: (Optional) Model filename for ``audio-separator`` (e.g., ``htdemucs_6s.yaml``).
- ``--model_file_dir``: (Optional) Directory containing ``audio-separator`` model files.

Example Output
==============

After running the tool, your output directory will contain the separated audio stems as WAV files and the combined plots as PNG images:

.. code-block::

   output_stems/
   ├── your_audio_bass.wav
   ├── your_audio_drums.wav
   ├── your_audio_other.wav
   ├── your_audio_vocals.wav
   ├── your_audio_combined_waveforms.png
   └── your_audio_combined_spectrograms.png


System Architecture
===================

The application is designed with a modular architecture, separating concerns into distinct components:

- **Command-Line Interface (CLI):** ``cli.py`` serves as the entry point for the application. It uses Python's ``argparse`` module to handle command-line arguments, allowing users to specify the input audio file or directory, the output directory, and the desired separation model.

- **Model Management:** ``models.py`` is responsible for loading and managing the audio separation models. It supports two main model backends:
    - **Torchaudio:** Utilizes the ``torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`` model, a pre-trained model for high-quality music source separation. This model separates audio into four stems: vocals, drums, bass, and other.
    - **audio-separator:** A library that provides a convenient interface to various source separation models, including different versions of Demucs. This allows for more flexibility, including the ability to separate into six stems (e.g., vocals, drums, bass, other, piano, guitar).

- **Audio Processing:** ``audio_processor.py`` contains the core logic for audio processing. The ``StemSeparator`` class orchestrates the entire process, from loading the audio file to applying the chosen model and saving the separated stems as WAV files. It handles audio resampling and format conversion to ensure compatibility with the models.

- **Plotting and Visualization:** ``plotter.py`` is responsible for generating visualizations of the separated audio. The ``AudioPlotter`` class creates two types of plots:
    - **Waveform Plots:** A combined plot showing the waveforms of all separated stems.
    - **Spectrogram Plots:** A combined plot showing the spectrograms of all separated stems.
    These plots are saved as PNG images in the output directory.

- **Configuration:** ``config.py`` centralizes all configuration parameters for the application, such as supported audio formats, default model settings, and color palettes for plotting.

Core Functionality
==================

Audio Separation
----------------

The audio separation process is initiated by the ``StemSeparator.separate_stems`` method. The steps are as follows:

1. **Audio Loading:** The input audio file is loaded using ``torchaudio.load``. The audio is resampled to the model's expected sample rate (44100 Hz) and converted to stereo if necessary.

2. **Model Inference:**
   - For the **Torchaudio** model, the waveform is passed through the pre-trained HDemucs model, which returns a tensor containing the separated stems.
   - For the **audio-separator** model, the ``Separator.separate`` method is called, which handles the separation process internally.

3. **Stem Saving:** Each separated stem is saved as a WAV file in the specified output directory.

Visualization
-------------

After the stems are separated, the ``AudioPlotter`` class generates visualizations:

1. **Waveform Plotting:** The ``plot_waveforms`` method loads each stem's WAV file, downsamples the waveform for efficient plotting, and generates a combined plot with each stem in its own subplot.

2. **Spectrogram Plotting:** The ``plot_spectrograms`` method computes the Short-Time Fourier Transform (STFT) of each stem's waveform using ``librosa.stft`` to generate a spectrogram. The spectrograms are then plotted in a combined image.

Dependencies
============

The application relies on the following key libraries:

- **torch & torchaudio:** For deep learning model inference and audio file I/O.
- **audio-separator:** For using a wider range of source separation models.
- **matplotlib & seaborn:** For generating high-quality plots.
- **librosa:** For advanced audio analysis, specifically for generating spectrograms.
- **numpy:** For numerical operations on audio data.
