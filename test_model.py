import time
import numpy as np
import pandas as pd
import pickle

from numpy.linalg import norm

from hazm import *
from hazm.lemmatizer import Lemmatizer
from hazm.normalizer import Normalizer
from hazm.pos_tagger import POSTagger
from hazm.word_tokenizer import WordTokenizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# Functions

def print_log_start(mission):
    print(mission + '...')
    return time.time()


def print_log_end(mission, start):
    print('Total ' + mission, time.time() - start)
    pass


def get_doc_vector(tokenized_doc, dtm_key_list):
    dtm_len = len(dtm_key_list)
    vector = [0 for j in range(dtm_len)]
    for i in range(dtm_len):
        vector[i] += tokenized_doc.count(dtm_key_list[i])
    return vector


def perform_pca_to_test_data(test_dtm, principal_test):
    test_data_pca = principal_test.transform(test_dtm)
    # Check the dimensions of data after PCA
    print(test_data_pca.shape)
    return test_data_pca


def get_cosine_distance(x1, x2):
    cosine = np.dot(x1, x2) / (norm(x1) * norm(x2))
    return cosine


def euclidean_distances(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def get_f_score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def knn_predict(test_data, k):
    distances = [euclidean_distances(test_data, x) for x in X_train]
    # cosine_distance more precised
    # distances = [get_cosine_distance(test_data, x) for x in X_train]
    sorted_distances = np.argsort(distances)[:k]
    k_nearest = [y_train[i] for i in sorted_distances]
    if k_nearest.count('Sport') > k_nearest.count('Politics'): return 'Sport'
    return 'Politics'


def knn_score(test_data, test_labels, k):
    # Positive for sport, Negative for politics

    tp, fp, fn = 0, 0, 0

    correct_predictions = 0
    #
    answers = []
    #
    for i in range(len(test_data)):
        predict = knn_predict(test_data[i], k)
        #
        answers.append(predict)
        #
        if (predict == test_labels[i]):
            correct_predictions += 1
            # True for sport
            if (predict == 'Sport'):
                tp += 1
        else:
            if predict == 'Sport':
                fp += 1
            else:
                fn += 1
    print(answers)
    normal_accuracy = correct_predictions / len(test_data)
    f1 = get_f_score(tp, fp, fn)
    # return normal_accuracy
    return f1


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
    # print(tagged_tokenized_news)
    news_tokenized_str = ''
    for word in tagged_tokenized_news:
        if all(word_type not in word[1] for word_type in forbidden_word_types) and ('!' not in word[0]) and (
                '?' not in word[0]):
            lemmatized_word = lemmatizer.lemmatize(word[0])
            clean_tagged_tokenized_news.append(lemmatized_word)
            # news_tokenized_str += lemmatized_word + ' '

    # Log : Cleaned tokenized news
    print(clean_tagged_tokenized_news)
    return clean_tagged_tokenized_news


# Load train data
df = pd.read_csv('resources/nlp_train.csv')
first_df = df
df = df[1000:19000]
data = df.iloc[:, 1:2]

#### test with input dataset


filename = 'train_files/finalized_data_pca.sav'
data_pca = pickle.load(open(filename, 'rb'))

X_train = data_pca
y_train = np.array(df.iloc[:, 2])





input_df = pd.read_csv('test/nlp_test.csv')
input_df = input_df[:1000]
y_test = np.array(input_df.iloc[:, 2])

test_docs = []
## Log #############
start_time = print_log_start("Tokenizing test words")
#####################

# Tokenize test data
for x in input_df['Text']:
    test_docs.append(tokenize_news(x))

## Log #############
print_log_end("Tokenizing test words", start_time)
#####################

## Log #############
start_time = print_log_start("Create test DTM")
#####################


# ver--4
dtm_keys_file_name = 'train_files/dtm_keys.sav'

dtm_keys = pickle.load(open(dtm_keys_file_name, 'rb'))

test_dtm_array = []
for i in range(len(test_docs)):
    test_dtm_array.append(get_doc_vector(test_docs[i], dtm_keys))

test_dtm = pd.DataFrame(test_dtm_array)

## Log #############
print_log_end("Create test DTM ", start_time)
#####################


## Log #############
start_time = print_log_start("PCA Test")
#####################

# ver--4
pca_mapping_file_name = 'train_files/mapping_pca.sav'
# ver--1
# pca_mapping_file_name = 'backup_pickle_saved_ver--1_n_component_1000/mapping_pca.sav'
principal_test = pickle.load(open(pca_mapping_file_name, 'rb'))

## Log #############
print_log_end("PCA Test", start_time)
#####################


X_test = np.array(perform_pca_to_test_data(test_dtm, principal_test))

# KNN

## Log #############
start_time = print_log_start("KNN")
#####################


# print("mine", knn_score(X_test, y_test, 9))

# best k : 15
print("mine", knn_score(X_test, y_test, 21))

## Log #############
print_log_end("KNN", start_time)
#####################

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=23)
# Fit the classifier to the data
knn.fit(X_train, y_train)
print("their", knn.score(X_test, y_test))

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)
