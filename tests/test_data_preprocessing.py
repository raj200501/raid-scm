import unittest
import pandas as pd
from src.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def test_preprocess_data(self):
        data = {'summary': ['This is a great day.', 'This is a terrible day.'], 'sentiment': ['positive', 'negative']}
        df = pd.DataFrame(data)
        X_train, X_test, y_train, y_test, _ = preprocess_data(df)
        self.assertEqual(len(X_train), 1)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_train), 1)
        self.assertEqual(len(y_test), 1)

if __name__ == '__main__':
    unittest.main()
