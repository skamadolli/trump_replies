import tensorflow as tf
import corpusload
import math
import numpy as np
import pickle
from scipy import spatial

'''
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
'''


word_to_ids = {}
final_embeddings = ""

count = []

traindata = corpusload.load_corpus('Source.txt', count, word_to_ids)

vocab_size = len(word_to_ids)
batch_size = 128

skip_size = 4
skip_window = 2

edim = 100
num_negative = 64

center = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None, 1])

embeddings = tf.Variable(tf.random_normal([vocab_size, edim]))
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
embed = tf.nn.embedding_lookup(embeddings, center)

nce_weights = tf.Variable(tf.truncated_normal([vocab_size, edim], stddev=1.0/math.sqrt(edim)))
nce_biases = tf.Variable(tf.zeros(vocab_size))
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=target,
                   inputs=embed,
                   num_sampled=num_negative,
                   num_classes=vocab_size))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_epochs = 50
    for epoch in range(num_epochs):
        for iter, index in enumerate(np.arange(skip_window, len(traindata) - skip_window - batch_size/skip_size, batch_size/skip_size)):
            batches, labels = corpusload.generate_batch(traindata, index, batch_size, skip_size, skip_window)
            batch_dict = {center: batches,
                          target: labels}
            sess.run(optimizer, feed_dict=batch_dict)
        print('EPOCH:   ', epoch+1,'/', num_epochs)

    embed_normalized = normalized_embeddings.eval()
    embedding_matrix = embeddings.eval()
    embed_weights = nce_weights.eval()
    embed_biases = nce_biases.eval()

#store the generated embedding so we can use it for a layer in the LSTM
with open('embeddings.pkl', 'wb') as f:
    pickle.dump([embed_normalized, embedding_matrix, embed_weights, embed_biases    , word_to_ids], f)


def closest_words_test(word, embeddings, word_to_ids):
    reversed_dict = dict(zip(word_to_ids.values(), word_to_ids.keys()))
    list = np.arange(0, embeddings.shape[0])
    remlist1 = list[:word_to_ids[word]]
    remlist2 = list[(word_to_ids[word]+1):]
    rememb = np.concatenate((embeddings[remlist1], embeddings[remlist2]), axis = 0)
    tree = spatial.KDTree(rememb)
    dist, index = tree.query(embeddings[word_to_ids[word]])
    if index >= word_to_ids[word]:
        index += 1
    return dist, reversed_dict[index]

d, w1 = closest_words_test('before', embedding_matrix, word_to_ids)
print('before - closest word ', w1)
