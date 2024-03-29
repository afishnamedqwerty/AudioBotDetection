import librosa
import numpy as np
import webrtcvad
from pyAudioAnalysis import audioSegmentation as aS

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
