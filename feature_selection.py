from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import numpy as np
from datasets import load_dataset
import librosa


# Extract Mel-frequency cepstrum coefficients from audio
def extract_cepstral_coefficients(audio, sr=16000, n_mfcc=20, lifter=0):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio signal.
    
    Parameters:
    - audio: The audio signal from which to extract features.
    - sr: The sample rate of the audio signal.
    - n_mfcc: The number of MFCCs to extract.
    - lifter: The liftering coefficient to apply. Liftering can help emphasize higher-order coefficients.
              Set to 0 to disable liftering.
    
    Returns:
    - An array of MFCCs averaged across time.
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, lifter=lifter)
    return np.mean(mfccs.T,axis=0)

# Extract Chroma STFT features from audio
def extract_chroma_stft(audio, sr=16000):
    stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    return np.mean(stft.T,axis=0)

# Extract Pitch
def extract_pitch(audio, sr=16000, fmin=75, fmax=300):
    pitches, _ = librosa.piptrack(y=audio, sr=sr, fmin=fmin, fmax=fmax)
    pitch = np.mean(pitches[pitches > 0])
    return np.array([pitch if pitch > 0 else 0])

# Extract Jitter
def extract_jitter(audio, sr=16000):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    jitter = np.abs(np.diff(pitches[pitches > 0])).mean()
    return np.array([jitter if not np.isnan(jitter) else 0])

# Load GigaSpeech dataset subset
def load_gigaspeech_dataset(subset='xs', use_auth_token=True):
    """
    Load a specified subset of the GigaSpeech dataset.
    """
    gs = load_dataset("speechcolab/gigaspeech", subset, use_auth_token=use_auth_token)
    return gs

# Extract selected features from the dataset
def extract_features(gs, feature_funcs):
    """
    Extract features from the GigaSpeech dataset.

    Parameters:
    - gs: The GigaSpeech dataset.
    - feature_funcs: A dictionary of functions to apply for feature extraction.
    
    Returns:
    - An array of extracted features from the dataset.
    """
    features = []
    for i in range(len(gs["train"])):
        audio_input = gs["train"][i]["audio"]["array"]
        feature_row = np.hstack([func(audio_input) for func in feature_funcs.values()])
        features.append(feature_row)
    return np.array(features)

# Perform PCA to reduce the dimensionality of the feature set
def perform_pca(X, n_components=0.95):
    """
    Perform PCA for dimensionality reduction.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

# Select the top features based on univariate statistical tests
def select_top_features(X, y, num_features=20):
    """
    Selects the top 'num_features' based on univariate statistical tests.
    """
    fs = SelectKBest(score_func=f_classif, k=num_features)
    fs.fit(X, y)
    X_selected = fs.transform(X)
    return X_selected, fs

# After feature extraction and selection
def reshape_features_for_lstm(features, num_features_per_timestep):
    """
    Reshape the 2D features array (samples, features) into a 3D array (samples, timesteps, features_per_timestep)
    suitable for LSTM input. This example assumes each sample is a single timestep.
    """
    samples = features.shape[0]
    timesteps = 1  # Assuming each sample is one timestep; adjust as necessary.
    features_per_timestep = num_features_per_timestep  # Adjust based on your feature selection

    return features.reshape((samples, timesteps, features_per_timestep))

'''def select_features(X_train, y_train, X_test, num_features=10):
    """
    Selects the top 'num_features' based on univariate statistical tests.
    """
    fs = SelectKBest(score_func=f_classif, k=num_features)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs'''

'''def extract_features(audio_path, features_to_use=['mfcc', 'chroma', 'mel']):
    y, sr = librosa.load(audio_path, sr=None)
    features = {}
    
    if 'mfcc' in features_to_use:
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        features['mfcc'] = np.mean(mfccs, axis=1)
    
    if 'chroma' in features_to_use:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma'] = np.mean(chroma, axis=1)
    
    if 'mel' in features_to_use:
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features['mel'] = np.mean(mel, axis=1)
    
    return features'''


'''Tailored Detection Approach for Audio Analysis'''

'''
    Prosodic features: Rhythm, stress, and intonation patterns. 
    These can indicate natural speech flow and emotional nuances 
    typical of human speech.

    Spectral Features: Include MFCCs, formants, and spectral contrast. 
    These features capture the timbre and frequency distribution, 
    which differe between human and synthetic speech.

    Voice Quality Features: Harmonics-to-noise ratio (HNR) and jitter, 
    which can indicate the naturalness and stability of the voice.

    Principal Component Analysis: Reduce dimensionality of the feature 
    set to retain the most informative aspects for distinguishing 
    between human and synthesized voices.
'''