import numpy as np
import pickle
import tensorflow as tf
from rnn import helper
import datetime


def load_files():
    words_with_sentiments = np.load('data/tweet_words_with_sentiment_matrix.zip', allow_pickle=True)[
        'tweet_words_with_sentiment_matrix']
    ids = np.load('data/tweet_word_ids_matrix.zip')['tweet_words_ids_matrix']
    with open('data/embeddings.pkl', 'rb') as handle:
        embed = pickle.load(handle)
    return words_with_sentiments, ids, embed


batch_size = 24
lstm_units = 64
num_classes = 2
iterations = 100000

max_tweet_words = 20
num_dimensions = 100


def build_rnn_model():
    tf.reset_default_graph()
    words_with_sentiments, ids, embed = load_files()

    labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    input_data = tf.placeholder(tf.int32, [batch_size, max_tweet_words])

    data = tf.Variable(tf.zeros([batch_size, max_tweet_words, num_dimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(ids, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    train(ids, optimizer, input_data, labels, loss, accuracy)


def train(ids, optimizer, input_data, labels, loss, accuracy):
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    sess = tf.InteractiveSession()
    writer = tf.summary.FileWriter(logdir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = helper.get_train_batch(batch_size, max_tweet_words, ids)
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if i % 50 == 0:
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess, "data/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)

    writer.close()


if __name__ == '__main__':
    build_rnn_model()
