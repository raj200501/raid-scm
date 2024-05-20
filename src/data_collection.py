import requests
from bs4 import BeautifulSoup
import pandas as pd

def web_scraping(url):
    """
    Scrape data from the given URL.

    Args:
        url (str): URL of the website to scrape.

    Returns:
        pd.DataFrame: DataFrame containing headlines and summaries.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = []
    for article in soup.find_all('div', class_='article'):
        headline = article.find('h2').text.strip()
        summary = article.find('p').text.strip()
        data.append({'headline': headline, 'summary': summary})
    return pd.DataFrame(data)
