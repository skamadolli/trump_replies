from random import randint
import numpy as np

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
max_tweet_words = 20


def get_train_batch(ids):
    labels = []
    arr = np.zeros([batchSize, max_tweet_words])
    for i in range(batchSize):
        if i % 2 == 0:
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def get_test_batch(ids):
    labels = []
    arr = np.zeros([batchSize, max_tweet_words])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if num <= 12499:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels
