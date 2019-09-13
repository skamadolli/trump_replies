import tensorflow as tf
from rnn import helper
from rnn import rnn_lstm

batch_size = 24
max_tweet_words = 20
num_classes = 2
lstm_units = 64
num_dimensions = 100


def run_test_batches():
    tf.reset_default_graph()

    words_with_sentiments, word_indices, embedding_matrices, \
    labels, input_data, data, lstmCell, \
    weight, bias, value, last, prediction, correct_pred, \
    accuracy, loss, optimizer = rnn_lstm.build_rnn_model()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('data/trained_model'))
        iterations = 10
        for i in range(iterations):
            nextBatch, nextBatchLabels = helper.get_test_batch(batch_size, max_tweet_words, word_indices)
            test_acc = sess.run(accuracy, feed_dict={input_data: nextBatch, labels: nextBatchLabels})
            print("Batch Accuracy:", test_acc)


if __name__ == '__main__':
    run_test_batches()
