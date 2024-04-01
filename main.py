main.py

from feature_selection import *
from model import build_model, train_model, predict, evaluate_model
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # Load dataset subset
    gs = load_gigaspeech_dataset(subset="xs", use_auth_token=True)
    feature_funcs = {
        'mfccs': extract_mfccs,
        'chroma': extract_chroma_stft,
        'pitch': extract_pitch,
        'jitter': extract_jitter,
        'hnr': extract_hnr
        # Add additional feature extraction functions as needed
    }

    features = extract_features(gs, feature_funcs)
    if features is None:
        print("Failed to extract features - exiting...")
        return
    
    # Load labels from file or dataset
    try:
        labels = np.load("path/to/labels.npy")
    except Exception as e:
        print("Labels file not found. Exiting...")
        return
    # Assuming 'labels' are binary labels indicating human (0) or synthesized (1) voices
    #labels = np.random.randint(2, size=features.shape[0])  # Placeholder for actual labels

    # Update to load actual labels indicating human (0) or synthesized (1) voices
    #labels = np.load("path/to/labels.npy")  # Assuming labels are stored in a NumPy array

    
    # Perform PCA and feature selection
    features_pca = perform_pca(features)
    features_selected, _ = select_top_features(features_pca, labels, num_features=20)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features_selected, labels, test_size=0.2, random_state=42)
    
    # Reshape features for LSTM
    lstm_feature_shape = reshape_features_for_lstm(X_train, X_train.shape[1])

    # Build and train the LSTM model
    model, history = train_model(lstm_feature_shape, y_train, X_val, y_val, epochs=10, batch_size=32)
    
    # Evaluate the model
    evaluate_model(model, X_val, y_val)

    print("Analysis complete.")




    #y_clean, sr = load_and_reduce_noise(file_path)
    #segments = diarize_audio(file_path)
    
    # Apply VAD
    #speech_frames = vad_audio(y_clean, sr)
    
    # Extract features
    #lstm_feature_shape = extract_mfcc(y_clean, sr)

    # Apply PCA to the extracted features for dimensionality reduction
    #features_pca, _ = perform_pca(features)

    #features_selected, _ = select_top_features(features_pca, labels, num_features=20)
    #lstm_feature_shape = reshape_features_for_lstm(features_selected, 20)
    
    
    # Build and compile the LSTM model
    #model = build_model(input_shape=lstm_feature_shape.shape[1])

    # Assume X_train, y_train are prepared and available
    #model.fit(X_train, y_train, epochs=10, batch_size=32)  # Fit the model on your dataset
    
    #print("Analysis complete.")

if __name__ == "__main__":
    main()
