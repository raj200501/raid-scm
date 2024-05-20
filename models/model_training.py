# model_training.ipynb

import pandas as pd
from src.data_preprocessing import preprocess_data
from src.predictive_modeling import build_model, train_model, evaluate_model
from src.config import load_config

# Load the data
df = pd.read_csv('data/scraped_data.csv')

# Load configuration
config = load_config()

# Preprocess data
X_train, X_test, y_train, y_test, _ = preprocess_data(df, config['training']['test_size'], config['training']['random_state'])

# Tokenize data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100)
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100)

# Build model
model = build_model(input_dim=len(tokenizer.word_index) + 1,
                    embedding_dim=config['model']['embedding_dim'],
                    lstm_units=config['model']['lstm_units'],
                    dense_units=config['model']['dense_units'],
                    dropout_rate=config['model']['dropout_rate'])

# Train model
history = train_model(X_train_pad, y_train, model, epochs=config['model']['epochs'], batch_size=config['model']['batch_size'])

# Evaluate model
evaluate_model(model, X_test_pad, y_test)
