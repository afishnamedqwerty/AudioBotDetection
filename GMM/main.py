import numpy as np
from feature_selection import extract_features, scale_features
from model import train_gmm, compute_eer, classify_samples
from utils import load_ljspeech_dataset
from sklearn.model_selection import train_test_split

def main():
    # Example placeholder for loading and preparing the dataset
    audios, texts = load_ljspeech_dataset()
    # In practice, you would load your dataset here
    features_real = np.random.rand(100, 20)  # Placeholder for real audio features
    features_synthetic = np.random.rand(100, 20)  # Placeholder for synthetic audio features
    labels = np.array([0]*100 + [1]*100)  # 0 for real, 1 for synthetic

    # Combine and split the dataset
    features = np.vstack((features_real, features_synthetic))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train GMMs
    gmm_real = train_gmm(X_train[y_train == 0], n_components=16)
    gmm_synthetic = train_gmm(X_train[y_train == 1], n_components=16)

    # Classify and evaluate
    scores = classify_samples(gmm_real, gmm_synthetic, X_test)
    eer = compute_eer(y_test, scores)
    print(f"Equal Error Rate (EER): {eer:.2f}")

if __name__ == "__main__":
    main()
