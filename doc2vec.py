#!/usr/bin/env python3
import nltk
from nltk.corpus import brown
from typing import List
import string
from math import ceil
import pickle

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

def generate_training_data(sentences: List[List[str]], context_len: int):
    all_words = sorted(set([w for s in sentences for w in s]))

    word_to_num = {w: i for i, w in enumerate(all_words)}

    X = []
    y = []

    def register_sample(w, context, isen):
        iw = word_to_num[w]
        icontext = [word_to_num[c] for c in context]

        X.append((isen, *icontext))
        y.append(iw)


    for (isen, s) in enumerate(sentences):
        for i in range(0, len(s) - context_len):
            context = s[i : i + context_len]
            w = s[i + context_len]
            register_sample(w, context, isen)

    return X, y, word_to_num


def build_model(nwords: int, nsentences: int, context_len: int):

    inputs = keras.layers.Input(
            shape=(nsentences + nwords * context_len, )
            )

    X = keras.layers.Dense(256)(inputs)

    outputs = keras.layers.Dense(nwords, activation='softmax')(X)

    model = keras.models.Model(inputs=inputs,
                               outputs=outputs,
                               name='D2V')

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
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

    context_len = 3
    X, y, word_to_num = generate_training_data(sentences, context_len)

    # with open('dict.bin', 'wb') as f:
    #     pickle.dump(word_to_num, f)

    X = np.array(X)
    y = np.array(y)

    nitems = X.shape[0]
    assert X.shape[0] == y.shape[0]

    nsentences = len(sentences)
    nwords = len(word_to_num.keys())

    model = build_model(nwords, nsentences, context_len)

    print(model.summary())


    def generate_batch(batch_size):
        at = 0

        indexes = np.arange(nitems)
        np.random.shuffle(indexes)

        while at < nitems:
            next = min(at + batch_size, nitems)

            chosen = indexes[at : next]

            paragraph_features = to_categorical(X[chosen][:, 0], num_classes=nsentences)
            context_features = np.reshape(
                    to_categorical(X[chosen][:, 1:], num_classes=nwords),
                    (batch_size, nwords * context_len)
                    )

            Xnow = np.hstack((paragraph_features, context_features))

            ynow = y[chosen]
            yield Xnow, ynow

            at = next
        return None


    batch_size = 64

    model.fit_generator(
        generate_batch(batch_size),
        steps_per_epoch=ceil(nitems / batch_size))

    model.save('doc2vec.h5')


if __name__ == '__main__':
    main()
