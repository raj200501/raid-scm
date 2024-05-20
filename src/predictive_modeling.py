import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def build_model(input_dim, embedding_dim=128, lstm_units=64, dense_units=64, dropout_rate=0.5):
    """
    Build and compile the neural network model.

    Args:
        input_dim (int): Input dimension for the embedding layer.
        embedding_dim (int): Dimension of the embedding vectors.
        lstm_units (int): Number of units in the LSTM layer.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.keras.Model: Compiled neural network model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=100),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, model, epochs=5, batch_size=32, validation_split=0.1):
    """
    Train the neural network model.

    Args:
        X_train (np.array): Training data.
        y_train (np.array): Training labels.
        model (tf.keras.Model): Compiled neural network model.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of the training data to be used as validation data.

    Returns:
        tf.keras.callbacks.History: Training history of the model.
    """
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def predictive_modeling(df, config):
    """
    Perform predictive modeling using the given configuration.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, config['training']['test_size'], config['training']['random_state'])
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100)

    model = build_model(input_dim=len(tokenizer.word_index) + 1,
                        embedding_dim=config['model']['embedding_dim'],
                        lstm_units=config['model']['lstm_units'],
                        dense_units=config['model']['dense_units'],
                        dropout_rate=config['model']['dropout_rate'])

    train_model(X_train_pad, y_train, model, epochs=config['model']['epochs'], batch_size=config['model']['batch_size'])

    evaluate_model(model, X_test_pad, y_test)
