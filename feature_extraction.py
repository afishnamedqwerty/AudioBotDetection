import librosa

def extract_mfcc(y, sr, n_mfcc=13):
    """Extract MFCC features from an audio signal."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs
