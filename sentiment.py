#!/usr/bin/env python3
import pickle
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

def main():
    model = keras.models.load_model('word2vec.h5')

    with open('dict.bin', 'rb') as f:
        words = pickle.load(f)

    weights = model.get_weights()[0].shape


    # X_neg = [movie_reviews.words(f) for f in movie_reviews.fileids('neg')]
    # X_pos = [movie_reviews.words(f) for f in movie_reviews.fileids('pos')]



if __name__ == '__main__':
    main()
