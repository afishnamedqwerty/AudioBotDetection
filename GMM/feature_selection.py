import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from python_speech_features import logfbank  # Assuming use of python_speech_features for LFCC

def extract_lfcc(audio, sr=16000, n_filters=26, n_lfcc=20):
    """
    Extracts Linear Frequency Cepstral Coefficients (LFCC) from an audio signal.

    Parameters:
    - audio: The audio signal from which to extract features.
    - sr: The sample rate of the audio signal.
    - n_filters: The number of filters to use in the filterbank.
    - n_lfcc: The number of LFCCs to extract.

    Returns:
    - lfcc_features: An array of LFCC features averaged across time.
    """
    # Compute log filterbank energies.
    logfbank_features = logfbank(audio, samplerate=sr, nfilt=n_filters)
    
    # Compute DCT to get LFCCs, keep first 'n_lfcc' coefficients
    lfcc_features = np.fft.dct(logfbank_features, type=2, axis=1, norm='ortho')[:, :n_lfcc]
    return np.mean(lfcc_features, axis=0)

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

# Extract Harmonic-to-Noise Ratio (HNR)
def extract_hnr(audio, sr=16000):
    """
    Extracts the Harmonic-to-Noise Ratio (HNR) from an audio signal.
    
    Parameters:
    - audio: The audio signal from which to extract HNR.
    - sr: The sample rate of the audio signal.
    
    Returns:
    - An array with the calculated HNR value.
    """
    harmonic, percussive = librosa.effects.hpss(audio)
    hnr = np.mean(librosa.effects.harmonic(audio))
    return np.array([hnr if not np.isnan(hnr) else 0])

def extract_features(audio_path, sr=16000):
    """
    Wrapper function to load an audio file, pre-process it, and extract relevant features.

    Parameters:
    - audio_path: Path to the audio file.
    - sr: Sample rate to use for loading the audio.

    Returns:
    - features: A numpy array containing extracted features.
    """
    audio, sr = librosa.load(audio_path, sr=sr)
    lfcc = extract_lfcc(audio, sr)
    return lfcc

def scale_features(features):
    """
    Scales the features using StandardScaler.

    Parameters:
    - features: Numpy array of features to scale.

    Returns:
    - scaled_features: Scaled features.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.reshape(-1, 1))
    return scaled_features.flatten()

# Modified extract_features function to utilize parallel processing
def extract_features(gs, feature_funcs):
    """
    Extract features from the GigaSpeech dataset using parallel processing.
    
    Parameters:
    - gs: The GigaSpeech dataset.
    - feature_funcs: A dictionary of functions to apply for feature extraction.
    
    Returns:
    - An array of extracted features from the dataset.
    """
    features = []
    
    def process_audio(audio_input):
        return np.hstack([func(audio_input) for func in feature_funcs.values()])
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_audio, gs["train"][i]["audio"]["array"]) for i in range(len(gs["train"]))]
        for future in futures:
            features.append(future.result())
    
    return np.array(features)


# Extract selected features from the dataset - Deprecated
'''def extract_features(gs, feature_funcs):
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
    return np.array(features)'''

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
