import numpy as np
import pickle
import tensorflow as tf
from rnn import helper


def load_files():
    ids = np.load('data/tweet_word_ids_matrix.zip')['tweet_words_ids_matrix']
    with open('data/embeddings.pkl', 'rb') as handle:
        embed = pickle.load(handle)
    return ids, embed


def build_rnn_model():
    (ids, embed) = load_files()
    tf.reset_default_graph()

    num_of_tweets = len(ids)

    labels = tf.placeholder(tf.float32, [helper.batchSize, helper.numClasses])
    input_data = tf.placeholder(tf.int32, [helper.batchSize, num_of_tweets])


if __name__ == '__main__':
    load_files()
