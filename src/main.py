from src.data_collection import web_scraping
from src.sentiment_analysis import sentiment_analysis
from src.topic_generation import topic_generation
from src.predictive_modeling import predictive_modeling
from src.config import load_config

def main():
    config = load_config()
    url = config['data']['url']
    df = web_scraping(url)
    df = sentiment_analysis(df)
    df = topic_generation(df)
    predictive_modeling(df, config)

if __name__ == "__main__":
    main()
