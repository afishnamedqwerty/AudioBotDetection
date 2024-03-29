# Audio Bot Detection

## Overview

This project aims to detect bot-generated audio in multi-speaker recordings from platforms such as Twitter Spaces. Utilizing advanced audio processing and machine learning techniques, the project identifies characteristics unique to synthetic voices, distinguishing them from human speakers.

## Components

- feature_selection.py: Extraction of audio features relevant for distinguishing human from bot-generated speech.
- model.py: A machine learning model trained to classify audio as either human or bot-generated.
- main.py: The main script orchestrating the analysis pipeline.
- utils.py: Noise reduction, audio diarization in test data, VAD, and additional audio utils.

## Preqrequisites

- Python 3.6 or higher
- Libraries: librosa, noisereduce, pyAudioAnalysis, webrtcvad, tensorflow, scikit-learn

## Installation

First, ensure that you have Python installed on your system. Then, install the required libraries using pip:

`pip install librosa noisereduce pyAudioAnalysis webrtcvad tensorflow scikit-learn`

For pyAudioAnalysis, you may need to follow additional installation instructions from its GitHub repository.

## Usage

To train the model:

`python main.py"`

## Project Structure
An LSTM model trained on GigaSpeech dataset; fit to audio-sampling features such as cepstral coefficients, chroma stft, pitch, and jitter to perform binary classification between human and AI-generative audio sampling. 

audio_analysis_project/
│
├── preprocessing.py       # Noise reduction and initial audio preparation
├── diarization.py         # Speaker diarization
├── vad.py                 # Voice activity detection
├── feature_extraction.py  # Audio feature extraction
├── model.py               # Machine learning model for classification
└── main.py                # Main script orchestrating the analysis pipeline

## License

This project is licensed under the BOTperative Corp (subsidiary of Tyrell Corporation) - see the LICENSE.md for details.

## Acknowledgements

- `librosa` for audio processing
- `pyAudioAnalysis` for speaker diarization support
- `webrtcvad` for voice activity detection
- `tensorflow` and `scikit-learn` for building and training the machine learning model