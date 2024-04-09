utils.py

import librosa
import webrtcvad
from pyAudioAnalysis import audioSegmentation as aS
import tensorflow_datasets as tfds
import numpy as np
import soundfile as sf
import os
import requests
from tqdm import tqdm

def load_ljspeech_dataset():
    """
    Load the LJ Speech dataset and prepare it for feature extraction.

    Returns:
    - audios: Numpy array of audio waveforms.
    - texts: List of corresponding normalized text transcriptions.
    """
    dataset, ds_info = tfds.load('ljspeech', split='train', with_info=True)
    audios = []
    texts = []

    for example in tfds.as_numpy(dataset):
        audio = example['speech'].astype(np.float32) / 32768.0  # Normalize int16 to float32 range [-1, 1]
        text = example['text_normalized']
        audios.append(audio)
        texts.append(text)

    return audios, texts

# Hypothetical function to download datasets
def download_dataset(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

# Example usage - replace with actual dataset URLs and paths
# download_dataset("https://example.com/ljspeech.tar.gz", "ljspeech.tar.gz")
# download_dataset("https://example.com/jsut.zip", "jsut.zip")
# ... code to extract the archives ...

# Directory where the audio files are stored
audio_dir = "path_to_audio_files"

# Pre-processing parameters
sample_rate = 22050  # Target sample rate
duration = 5  # Target duration in seconds

def preprocess_audio(file_path, target_sample_rate, target_duration):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=target_sample_rate)
    # Calculate target length
    target_length = target_sample_rate * target_duration
    # Pad audio if necessary
    if len(audio) < target_length:
        pad_length = target_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    # Trim audio if necessary
    elif len(audio) > target_length:
        audio = audio[:target_length]
    return audio

# Pre-process and save all audio files
for file_name in os.listdir(audio_dir):
    file_path = os.path.join(audio_dir, file_name)
    if file_path.endswith(".wav"):
        audio = preprocess_audio(file_path, sample_rate, duration)
        # Save the pre-processed audio file
        processed_file_path = os.path.join(audio_dir, "processed", file_name)
        sf.write(processed_file_path, audio, sample_rate)

print("Dataset preparation completed.")

import noisereduce as nr
import librosa

def noise_reduce(audio, sr):
    # Load an audio file. Replace 'path_to_your_audio_file.wav' with your actual audio file path.
    y, sr = librosa.load('path_to_your_audio_file.wav')

    # Perform noise reduction to clean up the audio from background noise.
    # This step is crucial for ensuring that the voice signals are clear for further analysis.
    y_clean = nr.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)

    return y_clean




def diarize_audio(file_path):
    """Perform speaker diarization on an audio file."""
    segments = aS.speaker_diarization(file_path, n_speakers=0, mid_window=1.0, mid_step=0.1, short_window=0.05, lda_dim=35, plot_res=False, sm_segment=True)
    return segments

def load_audio_file(file_path, sr=16000):
    """
    Loads an audio file at the specified sample rate.
    """
    y, _ = librosa.load(file_path, sr=sr)
    return y

def normalize_audio(audio):
    """
    Normalize audio signal to have zero mean and unit variance.
    """
    normalized_audio = (audio - np.mean(audio)) / np.std(audio)
    return normalized_audio

def compute_accuracy(y_true, y_pred):
    """
    Compute the accuracy of predictions.
    """
    accuracy = sum(y_true == y_pred) / len(y_true)
    return accuracy

def vad_audio(y, sr, frame_duration=10):
    """Voice activity detection."""
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    y_mono = librosa.to_mono(y)
    y_16bit = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)
    frame_length = int(16000 * frame_duration / 1000)
    frames = [y_16bit[i:i+frame_length] for i in range(0, len(y_16bit), frame_length)]
    is_speech = [vad.is_speech(frame.tobytes(), 16000) for frame in frames]
    return is_speech
