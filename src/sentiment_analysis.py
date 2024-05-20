from transformers import pipeline

def sentiment_analysis(df):
    """
    Perform sentiment analysis on the summaries in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing summaries.

    Returns:
        pd.DataFrame: DataFrame with an added sentiment column.
    """
    sentiment_model = pipeline('sentiment-analysis')
    df['sentiment'] = df['summary'].apply(lambda text: sentiment_model(text)[0]['label'])
    return df
