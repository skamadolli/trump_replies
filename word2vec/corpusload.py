import os
from collections import Counter
import numpy as np
import random


def load_corpus(fname, count, word2idx):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise ("Corpus file not found")

    words = []
    for line in lines:
        words.extend(line.split())

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)

    d = {};
    for key in words:
        d[key] = d.get(key, 0) + 1
    count.extend((sorted(d.items(), key = lambda x: x[1], reverse = True)))

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, a in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    corpus_idx = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            corpus_idx.append(index)
        corpus_idx.append(word2idx['<eos>'])

    print("Loaded corpus...")
    return corpus_idx

def generate_batch(data, index, batch_size, skip, window_size):
    batchindex = int(index)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    for i in range(batch_size // skip):
        center = data[batchindex]
        context = [w for w in range(2*window_size+1) if w != window_size]
        words_to_use = random.sample(context, skip)
        for j, word in enumerate(words_to_use):
            batch[i*skip + j] = center
            labels[i*skip + j] = data[batchindex + word - window_size]
        batchindex += 1
    return batch, labels
