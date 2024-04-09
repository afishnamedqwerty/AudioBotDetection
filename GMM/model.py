from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def train_gmm(features, n_components=16):
    """
    Train a Gaussian Mixture Model (GMM) on the provided features.

    Parameters:
    - features: Feature matrix for training the GMM.
    - n_components: Number of Gaussian components in the GMM.

    Returns:
    - gmm: Trained GMM object.
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=200, random_state=0)
    gmm.fit(features)
    return gmm

def compute_eer(y_true, y_scores):
    """
    Compute the Equal Error Rate (EER).

    Parameters:
    - y_true: Ground truth binary labels.
    - y_scores: Predicted scores or probabilities.

    Returns:
    - eer: Equal Error Rate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def classify_samples(gmm_real, gmm_synthetic, features):
    """
    Classify samples using trained GMMs based on the log-likelihood ratio.

    Parameters:
    - gmm_real: GMM trained on real audio features.
    - gmm_synthetic: GMM trained on synthetic audio features.
    - features: Feature matrix of samples to classify.

    Returns:
    - scores: Log-likelihood ratio scores for the samples.
    """
    log_likelihood_real = gmm_real.score_samples(features)
    log_likelihood_synthetic = gmm_synthetic.score_samples(features)
    scores = log_likelihood_real - log_likelihood_synthetic
    return scores
