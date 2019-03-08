#!/usr/bin/env python3
import random
import numpy as np
import swifter
import tqdm
from bs4 import BeautifulSoup
import string
import re
import pickle
import pandas as pd
import nltk
from typing import List
import string
from math import ceil

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

kMaxWords = 100000
kMaxSequenceLength = 40

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

    df['text'] = df['text'].swifter.apply(normalize)

    df.to_csv('datasets/clean_tweets.csv')

def filter_for_nans(df):
    def is_bad_fn(t):
        return type(t) != type('str')

    # In some rows, df['text'] is float instead of string.
    # Simply drop the bad rows.
    idx = np.argwhere(df['text'].apply(is_bad_fn)).flatten()
    df.drop(labels=idx, inplace=True)


def check_data(df):
    for t in df['text']:
        if type(t) != type('str'):
            print(t)
            print(type(t))
            assert type(t) == type('str')


def twitter_model():
    inputs = keras.Input(shape=(kMaxSequenceLength, ))

    X = inputs
    X = keras.layers.Embedding(kMaxWords, 32, input_length=kMaxSequenceLength)(X)
    X = keras.layers.LSTM(32)(X)
    X = keras.layers.Dense(2)(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='TwitterModel')

    model.summary()
    return model



def main():
    random.seed(1337)

    # make_clean_data()

    df = pd.read_csv('datasets/clean_tweets.csv')

    # filter_for_nans(df)
    # df.to_csv('datasets/clean_tweets.csv')

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=kMaxWords)
    tokenizer.fit_on_texts(df['text'])

    X = tokenizer.texts_to_sequences(df['text'])
    X = keras.preprocessing.sequence.pad_sequences(
            X,
            maxlen=kMaxSequenceLength,
            padding='post',
            truncating='post')
    y = np.array(df['sentiment'])
    y[y == 4] = 1
    y = keras.utils.to_categorical(y, num_classes=2)

    print(X.shape)
    print(y.shape)

    model = twitter_model()
    model.compile('adam', 'binary_crossentropy')

    model.fit(X, y)


if __name__ == '__main__':
    main()
