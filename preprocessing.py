import noisereduce as nr
import librosa

def load_and_reduce_noise(file_path):
    """Load an audio file and apply noise reduction."""
    y, sr = librosa.load(file_path)
    y_clean = nr.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)
    return y_clean, sr
