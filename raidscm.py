import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def web_scraping(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = []
    for article in soup.find_all('div', class_='article'):
        headline = article.find('h2').text.strip()
        summary = article.find('p').text.strip()
        data.append({'headline': headline, 'summary': summary})
    return pd.DataFrame(data)

def sentiment_analysis(df):
    sentiment_model = pipeline('sentiment-analysis')
    df['sentiment'] = df['summary'].apply(lambda text: sentiment_model(text)[0]['label'])
    return df

def topic_generation(df):
    tokenizer =GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    def generate_topic(text):
        inputs = tokenizer.encode(text, return_tensors='pt')
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    df['topic'] = df['summary'].apply(generate_topic)
    return df

def predictive_modeling(df):
    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
    X = df['summary'].values
    y = df['sentiment_encoded'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=100),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

def main():
    url = 'https://www.example.com/supply-chain-news'
    df = web_scraping(url)
    df = sentiment_analysis(df)
    df = topic_generation(df)
    predictive_modeling(df)

if __name__ == "__main__":
    main()
