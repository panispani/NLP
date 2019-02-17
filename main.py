#!/usr/bin/env python3
import nltk
from nltk.corpus import brown
from typing import List
import string
from math import ceil

import tensorflow as tf
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

    def register_sample(w, prev):
        w = word_to_num[w]
        prev = word_to_num[prev]

        X.append(w)
        y.append(prev)


    for s in sentences:
        for i in range(1, len(s)):
            w = s[i]
            prev = s[i - 1]
            register_sample(w, prev)

            # next = s[i + 1]
            # register_sample(w, prev, next)


    return X, y, word_to_num

def main():
    words = brown.words()

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

    model = Sequential()
    model.add(Dense(256, input_dim=nclasses))

    model.add(Dense(nclasses, activation='softmax'))
    # model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop',
                  #loss='sparse_categorical_crossentropy',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    def generate_batch(batch_size):
        at = 0

        indexes = np.arange(nitems)
        np.random.shuffle(indexes)

        while at < nitems:
            next = min(at + batch_size, nitems)

            chosen = indexes[at : next]

            Xnow = to_categorical(X[chosen], num_classes=nclasses)
            ynow = to_categorical(y[chosen], num_classes=nclasses)
            # ynow = y[chosen]

            yield Xnow, ynow

            at = next
        return None


    batch_size = 64

    model.fit_generator(
        generate_batch(batch_size),
        steps_per_epoch=ceil(nitems / batch_size))


if __name__ == '__main__':
    main()
