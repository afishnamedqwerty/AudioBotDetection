main.py

from preprocessing import load_and_reduce_noise
from diarization import diarize_audio
from vad import vad_audio
from feature_selection import extract_mfcc
from model import build_model
import feature_selection as fs
import numpy as np

def main():
    # Load dataset subset
    gs = fs.load_gigaspeech_dataset(subset="xs", use_auth_token=True)
    feature_funcs = {
        'cepstral_coeff': fs.extract_cepstral_coefficients,
        'chroma': fs.extract_chroma_stft,
        # Add additional feature extraction functions as needed
    }

    features = fs.extract_features(gs, feature_funcs)
    # Assuming 'labels' are binary labels indicating human (0) or synthesized (1) voices
    labels = np.random.randint(2, size=features.shape[0])  # Placeholder for actual labels
    
    # Perform PCA and feature selection
    features_pca = fs.perform_pca(features)
    features_selected, selector = fs.select_top_features(features_pca, labels)



    #y_clean, sr = load_and_reduce_noise(file_path)
    #segments = diarize_audio(file_path)
    
    # Apply VAD
    #speech_frames = vad_audio(y_clean, sr)
    
    # Extract features
    #lstm_feature_shape = extract_mfcc(y_clean, sr)

    # Apply PCA to the extracted features for dimensionality reduction
    features_pca, _ = fs.perform_pca(features)

    features_selected, _ = fs.select_top_features(features_pca, labels, num_features=20)
    lstm_feature_shape = fs.reshape_features_for_lstm(features_selected, 20)
    
    
    # Build and compile the LSTM model
    model = build_model(input_shape=lstm_feature_shape.shape[1])

    # Assume X_train, y_train are prepared and available
    model.fit(X_train, y_train, epochs=10, batch_size=32)  # Fit the model on your dataset
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
