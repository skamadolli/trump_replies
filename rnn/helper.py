from random import randint
import numpy as np


def get_train_batch(batch_size, max_tweet_words, ids):
    labels = []
    arr = np.zeros([batch_size, max_tweet_words])
    for i in range(batch_size):
        if i % 2 == 0:
            num = randint(1, 735999)
            labels.append([1, 0])  # negative
        else:
            num = randint(863999, 1599999)
            labels.append([0, 1])  # positive
        arr[i] = ids[num - 1:num]
    return arr, labels


def get_test_batch(batch_size, max_tweet_words, ids):
    labels = []
    arr = np.zeros([batch_size, max_tweet_words])
    for i in range(batch_size):
        num = randint(735999, 863999)
        if num <= 799999:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels
