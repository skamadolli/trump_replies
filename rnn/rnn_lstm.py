import numpy as np
import pickle
import tensorflow as tf
from rnn import helper


def load_files():
    words_with_sentiments = np.load('data/tweet_words_with_sentiment_matrix.zip', allow_pickle=True)[
        'tweet_words_with_sentiment_matrix']
    word_indices = np.load('data/tweet_words_ids_matrix.zip')['tweet_words_ids_matrix']
    with open('data/embeddings.pkl', 'rb') as handle:
        embedding_matrices = pickle.load(handle)
    return words_with_sentiments, word_indices, embedding_matrices


batch_size = 24
lstm_units = 64
num_classes = 2
iterations = 100000

max_tweet_words = 20
num_dimensions = 100


def build_rnn_model():
    tf.compat.v1.reset_default_graph()
    words_with_sentiments, word_indices, embedding_matrices = load_files()

    labels = tf.compat.v1.placeholder(tf.float32, [batch_size, num_classes])
    input_data = tf.compat.v1.placeholder(tf.int32, [batch_size, max_tweet_words])

    data = tf.Variable(tf.zeros([batch_size, max_tweet_words, num_dimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(embedding_matrices[1], input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    train(word_indices, optimizer, input_data, labels, loss, accuracy, prediction)


def train(word_indices, optimizer, input_data, labels, loss, accuracy, prediction):
    sess = tf.compat.v1.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(iterations):
        # Next Batch of tweets
        next_batch, next_batch_labels = helper.get_train_batch(batch_size, max_tweet_words, word_indices)
        sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})

        # Print to console every 1000
        if i % 1000 == 0 and i != 0:
            loss_ = sess.run(loss, {input_data: next_batch, labels: next_batch_labels})
            accuracy_ = sess.run(accuracy, {input_data: next_batch, labels: next_batch_labels})
            print("iteration {}/{}...".format(i + 1, iterations),
                  "loss {}...".format(loss_),
                  "accuracy {}...".format(accuracy_))

        # Save the network every 10,000 training iterations
        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess, "data/trained_model/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)


if __name__ == '__main__':
    build_rnn_model()
