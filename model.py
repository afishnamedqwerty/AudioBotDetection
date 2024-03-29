import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

'''Given the updated feature selections and considering the inclusion of PCA, the input to your model
    will now be a flattened vector of the selected features rather than raw audio data or spectrograms.
    This adjustment means you might not need the initial `Convo2D` layer designed for image-like input 
    (ex. spectrograms).'''

def build_model(input_shape, num_classes=2):
    """
    Build an LSTM model suitable for processing time-series features.
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    # Compile the model with Adam optimizer and cross-entropy loss
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    model = build_model((X_train.shape[1], X_train.shape[2], 1))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model, history

def predict(model, X):
    predictions = model.predict(X)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

'''Sigmoid limitations
    - Vanishing Gradient: When training deep neural networks, gradients can
    become very small during backpropogation, especially for inputs far from 
    0. This can lead to slow convergence or even completely halt the learning 
    process.

    -Output Saturation: Sigmoid outputs tend to saturate for very large or very 
    small input values, which means the gradient becomes close to zero. This can 
    also cause learning to slow down, particularly in the case of deep networks.

    - Not Zero-Centered: The sigmoid function is not zero-centered, which can make 
    optimization more challenging, especially when used in conjunction with certain 
    weight initialization methods'''

'''Softmax limitations
    - Multiclass Imbalance: Softmax is typically used in multi-class classification 
    tasks, but can lead to imbalance issues if there are many classes. The model may 
    become overly confident in its predictions for certain classes, especially when 
    the training data is imbalanced.

    - Lack of Robustness to Outliers: Softmax is sensitive to outliers in the input 
    data, as it relies on the enponentiation of input values. Outliers can 
    disproportionately influence the output probabilities, leading to less robust 
    predictions.

    - Dependency on Other Outputs: Softmax outputs are dependent on each other due 
    to the normalization step that ensures the sum of probabilities adds up to 1. 
    This can make it difficult to interpret individual output probabilities 
    independently of each other.'''