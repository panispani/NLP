#!/usr/bin/env python3
import random
from bs4 import BeautifulSoup
import string
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import movie_reviews
from typing import List
import string
from math import ceil

import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

import numpy as np

def get_word2vec():
    model = keras.models.load_model('word2vec.h5')
    vector_for_word = model.get_weights()[0]
    return vector_for_word

def make_clean_data():
    cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    df = pd.read_csv('./datasets/training.1600000.processed.noemoticon.csv',
            encoding='ISO-8859-1', header=None, names=cols)
    df.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

    wn = nltk.WordNetLemmatizer()

    def normalize(text):
        text = BeautifulSoup(text, 'lxml').get_text()
        text = re.sub(r'@[a-zA-z_0-9]+', '', text)
        text = re.sub(r'http(s)?://[a-zA-Z0-9./]+', '', text)
        text = re.sub(r"'".format(string.punctuation), '', text)
        text = re.sub(r'[{}]'.format(string.punctuation), ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[0-9]+', '', text)
        text = text.lower()
        text = ''.join([c for c in text if c in string.printable])
        text = wn.lemmatize(text)

        return text

    df['text'] = df['text'].apply(normalize)

    pd.options.display.max_colwidth = 100
    print(df.sample(frac=0.4))

    df.to_csv('datasets/clean_tweets.csv')


def main():
    random.seed(1337)

    with open('dict.bin', 'rb') as f:
        words = pickle.load(f)

    df = pd.read_csv('datasets/clean_tweets.csv')

    df['text'] = df['text']


if __name__ == '__main__':
    main()
