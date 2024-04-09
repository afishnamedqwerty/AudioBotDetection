# Audio Bot Detection

## Overview

!!! I need to clean this up, only relevant files are your respective dataset folder locations and the following components listed below.


This project aims to detect bot-generated audio in multi-speaker recordings from platforms such as Twitter Spaces. Utilizing advanced audio processing and machine learning techniques, the project focuses on extracting Linear Frequency Cepstral Coefficients (LFCC) from audio files and utilizing Gaussian Mixture Models (GMM) for the classification of real vs synthetic speech. The primary goal is to preprocess audio data, extract meaningful features, and apply machine learning models to distinguish between real and synthesized speech samples.

## Features

- **Dataset Downloading and Extraction**: Automated scripts to download and extract speech datasets from specified URLs.
- **Feature Extraction**: Computation of Mel Filterbanks and LFCC from audio signals.
- **Model Training and Evaluation**: Training of GMMs for real and synthetic voice samples and evaluation using the Equal Error Rate (EER) metric.
- **Visualization**: Visualizations of LFCC distributions and coefficients to analyze feature characteristics.

## Components

- `run.ipynb`: Dataset preprocessing, feature selection, model training & evaluation
- `batch_synthetic.py`: Generates a synthetic waveform dataest of audio files from the LJSPEECH dataset transcripts using the TACOTRON WAVERNN LJSPEECH pipeline.
- `test.ipynb`: CUDA munitions

## Preqrequisites

Officially tested on CUDA 11.8.1 w/ cuDNN 8.9.1
- Python 3.6 or higher (tested on 3.11.8)
- Pytorch 1.8.1 or higher (tested on 2.2.2)
- torchaudio 0.8.1 or higher (tested on 2.2.2)
- TensorFlow (for comparisons)
- Additional Libraries: 
- `os`
- `requests`
- `tqdm`
- `pathlib`
- `tarfile`
- `numpy`
- `librosa`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `soundfile`

## Usage

To train the model:

Use `run.ipynb` to download LJSPEECH dataset, run batch_synthetic.py to generate second synthetic dataset, contineu through `run.ipynb` for feature selection, model training, and evaluation.

## Audio Quality and Customization Options

### Sigma Tuning

The sigma parameter in WaveGlow inference controls the variance of the Gaussian distribution used for the audio data. Keep in mind this current project is configured for WAVERNN instead of Wave2Vec, which is required for sigma value. Adjusting sigma affects the audio quality:
- A lower sigma value leads to clearer audio with potential unnatural artifacts.
- A higher sigma value produces more natural audio with possible background noise.

### Model Selection

You can choose between character-based and phoneme-based processing for text input. Phoneme-based processing tends to produce more natural-sounding speech.

## Example

# Define your dataset URL and extraction directory
dataset_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
extract_to = "./ljs_dataset"

# Download and extract dataset
download_and_extract_dataset(dataset_url, extract_to)

# Load and preprocess audio files, then extract features
real_audio_dir = "./ljs_dataset/LJSpeech-1.1/wavs"
real_features = preprocess_and_extract_features([real_audio_dir])

synthetic_audio_dir = "./synthetic_ljs"
synthetic_features = preprocess_and_extract_features([synthetic_audio_dir])

# Train GMM models for real and synthetic voices (assuming synthetic features are obtained similarly)
real_gmm, synthetic_gmm = train_gmm_models(real_features, synthetic_features)

# Evaluate models and compute EER
EER, eer_threshold = evaluate_gmm(real_gmm, synthetic_gmm, test_features, true_labels)


SMA's


