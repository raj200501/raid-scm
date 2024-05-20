# EDA.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('data/scraped_data.csv')

# Basic statistics
print(df.describe())

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sentiment')
plt.title('Sentiment Distribution')
plt.show()
