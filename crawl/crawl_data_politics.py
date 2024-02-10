import requests
import lxml
import csv

import pandas as pd
from bs4 import BeautifulSoup


def get_url(full_href):
    return  'https://www.khabaronline.ir' + full_href[0: full_href.rindex('/')]

def get_content(url):
    temp_req = requests.get(url)
    temp_soup = BeautifulSoup(temp_req.text, "lxml")
    main_content = temp_soup.find('main').find('div', {'class': 'main-content'})
    news_title = main_content.find('div', {'class' : "item-title"}).find('h1').text
    # item summary was not considered due to over fitting
    news_text_paragraphs = main_content.find('div', {'class' : "item-body"}).find_all('p')
    news_content = ''
    for p in news_text_paragraphs:
        news_content += p.text + ' '
    return '\n' + news_title + '\n' + news_content + '\n'

def is_about_politics(url):
    temp_req = requests.get(url)
    if (temp_req.status_code != 200):
        return False
    temp_soup = BeautifulSoup(temp_req.text, "lxml")
    if temp_soup.find('main').find('div', {'class': 'main-content'}) is None : return False
    news_subject_service = temp_soup.find('main').find('div', {'class': 'main-content'}).find_all('li', {'class': 'breadcrumb-item'})[1].find('a').get('href')


    news_subject = news_subject_service[news_subject_service.rindex('/') + 1 : ]
    return (news_subject == 'Politics')


khabar_online_urls = []

first_archive = 'https://www.khabaronline.ir/page/archive.xhtml?mn=11&wide=0&dy=5&ms=0&pi=1&yr=1402&tp=1'
count = 0;
for i in range(1, 150):
    new_archive = 'https://www.khabaronline.ir/page/archive.xhtml?mn=11&wide=0&dy=5&ms=0&pi=' + str(i) + '&yr=1402&tp=1'
    all_req = req = requests.get(new_archive)
    soup = BeautifulSoup(req.text, "lxml")
    figures = soup.find(('section', {'id' : "box202"})).find_all('figure')
    for fig in figures:
       link = fig.find('a').get('href')
       khabar_online_urls.append(get_url(link))
       print(count)
       count+=1




tag = 'Politics'
index = 1
with open('politic_news.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Url', 'Text', 'Category'])
    for url in khabar_online_urls:
        print(index)
        print(url)
        content = get_content(url)
        print(content)
        writer.writerow([index, url, content, tag])
        index += 1

