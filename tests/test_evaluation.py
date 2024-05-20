import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.evaluation import evaluate_model

class TestEvaluation(unittest.TestCase):

    def test_evaluate_model(self):
        model = Sequential([
            Dense(10, activation='relu', input_shape=(100,)),
            Dense(3, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        X_test = np.random.random((10, 100))
        y_test = np.random.randint(3, size=(10,))
        evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    unittest.main()
