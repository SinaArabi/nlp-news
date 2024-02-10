import time
import numpy as np
import requests
import lxml
import pickle
import pandas as pd
from bs4 import BeautifulSoup
from hazm import *
from hazm.lemmatizer import Lemmatizer
from hazm.normalizer import Normalizer
from hazm.pos_tagger import POSTagger
from hazm.word_tokenizer import WordTokenizer

from numpy.linalg import norm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


#Functions

def print_log_start(mission):
    print(mission + '...')
    return time.time()
def print_log_end(mission, start):
    print('Total ' + mission, time.time() - start)
    pass


def add_to_doc_matrix(word, doc_num, count_docs):
    if word not in term_doc_matrix.keys():
        term_doc_matrix[word] = [0 for j in range(count_docs)]
    term_doc_matrix[word][doc_num] += 1


def get_doc_vector(tokenized_doc, dtm_key_list):
    dtm_len = len(dtm_key_list)
    vector = [0 for j in range(dtm_len)]
    for i in range(dtm_len):
        vector[i] += tokenized_doc.count(dtm_key_list[i])
    return vector



# Tokenize using hazm

normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()
posTagger = POSTagger(model='resources/pos_tagger.model')
forbidden_word_types = ['PUNCT', 'CCONJ', 'VERB', 'ADP', 'SCONJ', 'DET', 'NUM', 'PRON']

def tokenize_news(news_content):
    normalized_news = normalizer.normalize(news_content)
    normalized_news = normalizer.remove_specials_chars(normalized_news)
    tokenized_news = tokenizer.tokenize(normalized_news)
    tagged_tokenized_news = posTagger.tag(tokens=tokenized_news)
    clean_tagged_tokenized_news = []
    for word in tagged_tokenized_news:
        if all(word_type not in word[1] for word_type in forbidden_word_types) and ('!' not in word[0]) and (
                '?' not in word[0]):
            lemmatized_word = lemmatizer.lemmatize(word[0])
            clean_tagged_tokenized_news.append(lemmatized_word)


    # Log : Cleaned tokenized news
    print(clean_tagged_tokenized_news)
    return clean_tagged_tokenized_news


#Load train data
df = pd.read_csv('resources/nlp_train.csv')
first_df = df
df = df[1000:19000]
data = df.iloc[:, 1:2]



#term-document matrix

term_doc_matrix = {}

docs = []

## Log #############
start_time = print_log_start('Tokenizing words')
#####################

for x in data['Text']:
    docs.append(tokenize_news(x))

## Log #############
print_log_end('Tokenizing words', start_time)
#####################

## Log #############
start_time = print_log_start("Create DTM")
#####################

for i in range(len(docs)):
    for token in docs[i]:
        add_to_doc_matrix(token, i, len(data['Text']))

keysList = list(term_doc_matrix.keys())

#Save dtm keys
dtm_keys_file_name = 'train_files/dtm_keys.sav'
pickle.dump(keysList, open(dtm_keys_file_name, 'wb'))

dtm = pd.DataFrame.from_dict(term_doc_matrix)

## Log #############
print_log_end("Create DTM", start_time)
#####################


#PCA

## Log #############
start_time = print_log_start("PCA")
#####################

# Set the n_components=1000
principal = PCA(1000)
principal.fit(dtm)

# save the PCA fit mapping to disk
pca_mapping_file_name = 'train_files/mapping_pca.sav'
pickle.dump(principal, open(pca_mapping_file_name, 'wb'))

data_pca = principal.transform(dtm)

# Check the dimensions of data after PCA
print(data_pca.shape)

## Log #############
print_log_end("PCA", start_time)
#####################

# save the model to disk
start_time = print_log_start("saving pca model")
#####################
filename = 'train_files/finalized_data_pca.sav'
pickle.dump(data_pca, open(filename, 'wb'))

## Log #############
print_log_end('saving pca model', start_time)
#####################

