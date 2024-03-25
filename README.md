# Audio Bot Detection

## Overview

This project aims to detect bot-generated audio in multi-speaker recordings from platforms such as Twitter Spaces. Utilizing advanced audio processing and machine learning techniques, the project identifies characteristics unique to synthetic voices, distinguishing them from human speakers.

## Components

- Preprocessing.py: Noise reduction and initial audio preparation.
- Diarization.py: Separation of audio into individual speaker segments.
- VAD.py: Identification of speech segments within the audio.
- Feature_Extraction.py: Extraction of audio features relevant for distinguishing human from bot-generated speech.
- Model.py: A machine learning model trained to classify audio as either human or bot-generated.
- Main.py: The main script orchestrating the analysis pipeline.

## Preqrequisites

- Python 3.6 or higher
- Libraries: librosa, noisereduce, pyAudioAnalysis, webrtcvad, tensorflow, scikit-learn

## Installation

First, ensure that you have Python installed on your system. Then, install the required libraries using pip:

`pip install librosa noisereduce pyAudioAnalysis webrtcvad tensorflow scikit-learn`

For pyAudioAnalysis, you may need to follow additional installation instructions from its GitHub repository.

## Usage

To run the audio bot detection pipeline, navigate to the project directory and execute:

`python main.py --audio_file "path_to_your_audio_file.wav"`

Replace "path_to_your_audio_file.wav" with the path to your audio file.

## Project Structure
`audio_analysis_project/
│
├── preprocessing.py       # Noise reduction and initial audio preparation
├── diarization.py         # Speaker diarization
├── vad.py                 # Voice activity detection
├── feature_extraction.py  # Audio feature extraction
├── model.py               # Machine learning model for classification
└── main.py                # Main script orchestrating the analysis pipeline`

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

- `librosa` for audio processing
- `pyAudioAnalysis` for speaker diarization support
- `webrtcvad` for voice activity detection
- `tensorflow` and `scikit-learn` for building and training the machine learning model