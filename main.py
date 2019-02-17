#!/usr/bin/env python3
import nltk
from nltk.corpus import brown
from typing import List
import string
from math import ceil

import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

import numpy as np

def split_into_sentences(words: List[str]) -> List[List[str]]:
    ret = []
    acc = []
    for w in words:
        if w == '.':
            ret.append(acc)
            acc = []
        else:
            acc.append(w)
    return ret

def remove_punctuation(w: str) -> bool:
    return ''.join([c for c in w if c not in string.punctuation])

def generate_training_data(sentences: List[List[str]]):
    all_words = sorted(set([w for s in sentences for w in s]))

    word_to_num = {w: i for i, w in enumerate(all_words)}

    X = []
    y = []

    def register_sample(w, prevw, nextw):
        w = word_to_num[w]
        prevw = word_to_num[prevw]
        nextw = word_to_num[nextw]

        X.append(w)
        y.append((prevw, nextw))


    for s in sentences:
        for i in range(1, len(s) - 1):
            w = s[i]
            prevw = s[i - 1]
            nextw = s[i + 1]
            register_sample(w, prevw, nextw)


    return X, y, word_to_num


def build_model(nclasses):

    inputs = keras.layers.Input(shape=(nclasses, ))

    X = keras.layers.Dense(256)(inputs)

    c1 = keras.layers.Dense(nclasses, activation='softmax')(X)
    c2 = keras.layers.Dense(nclasses, activation='softmax')(X)

    model = keras.models.Model(inputs=inputs,
                               outputs=[c1, c2],
                               name='W2V')

    model.compile(optimizer='adam',
                  loss=[
                      'sparse_categorical_crossentropy',
                      'sparse_categorical_crossentropy'],
                  metrics=['accuracy'])

    return model


def all_words():
    words = []
    words.extend(brown.words())

    for f in nltk.corpus.gutenberg.fileids():
        words.extend(nltk.corpus.gutenberg.words(f))

    for f in nltk.corpus.reuters.fileids():
        words.extend(nltk.corpus.reuters.words(f))

    return words


def main():

    words = all_words()

    sentences = split_into_sentences(words)

    sentences = [
        [remove_punctuation(w) for w in s]
        for s in sentences
    ]

    sentences = [
        [w for w in s if w]
        for s in sentences
    ]

    sentences = [
        [w.lower() for w in s if w]
        for s in sentences
    ]

    wn = nltk.WordNetLemmatizer()
    sentences = [
        [wn.lemmatize(w) for w in s if w]
        for s in sentences
    ]

    X, y, word_to_num = generate_training_data(sentences)

    X = np.array(X)
    y = np.array(y)

    nitems = X.shape[0]
    assert X.shape[0] == y.shape[0]
    nclasses = len(word_to_num.keys())

    model = build_model(nclasses)

    print(model.summary())


    def generate_batch(batch_size):
        at = 0

        indexes = np.arange(nitems)
        np.random.shuffle(indexes)

        while at < nitems:
            next = min(at + batch_size, nitems)

            chosen = indexes[at : next]

            Xnow = to_categorical(X[chosen], num_classes=nclasses)
            ynow = y[chosen]

            yield Xnow, (ynow[:, 0], ynow[:, 1])

            at = next
        return None


    batch_size = 64

    model.fit_generator(
        generate_batch(batch_size),
        steps_per_epoch=ceil(nitems / batch_size))

    model.save('word2vec.h5')


if __name__ == '__main__':
    main()
