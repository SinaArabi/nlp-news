import requests
import lxml
import csv

import pandas as pd
from bs4 import BeautifulSoup


def get_url(full_href):
    return full_href[0: full_href.rindex('/')]

def get_content(url):
    temp_req = requests.get(url)
    temp_soup = BeautifulSoup(temp_req.text, "lxml")
    news_title = temp_soup.find('div', {'class': 'news-detail-text'}).find('h1').text
    news_text_paragraphs = temp_soup.find('div', {'class': 'news-text'}).find_all('p')
    news_content = ''
    for p in news_text_paragraphs:
        news_content += p.text + ' '
    return '\n' + news_title + '\n' + news_content + '\n'


url = 'https://www.varzesh3.com/sitemap/news'
req = requests.get(url)
soup = BeautifulSoup(req.text, "lxml")
all_url_tags = soup.find_all('url')

varzesh3_urls = []

for url_tag in all_url_tags:
    varzesh3_urls.append(get_url(url_tag.find('loc').text))

tag = 'Sport'
index = 1
with open('sport_news.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Url', 'Text', 'Category'])
    for url in varzesh3_urls[0:2]:
        print(index)
        print(url)
        content = get_content(url)
        print(content)
        writer.writerow([index, url, content, tag])
        index += 1
