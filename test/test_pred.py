import numpy as np
import pickle
import tensorflow as tf
from rnn import helper
import datetime
import re



def load_files():
    words_with_sentiments = np.load('data/tweet_words_with_sentiment_matrix.zip', allow_pickle=True)[
        'tweet_words_with_sentiment_matrix']
    word_indices = np.load('data/tweet_words_ids_matrix.npy')
    with open('data/embeddings.pkl', 'rb') as handle:
        embedding_matrices = pickle.load(handle)
        #print(embedding_matrices[4])


    # for i in range(5):
    #     print("words_with_sentiments: ", words_with_sentiments[i])
    #     print("word_indices: ", word_indices[i])
    #     print("embedding: ", embedding_matrices[4])

    return words_with_sentiments, word_indices, embedding_matrices


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
batch_size = 24
lstm_units = 64
num_classes = 2
iterations = 5000

max_tweet_words = 20
num_dimensions = 100


def build_rnn_model():
    tf.compat.v1.reset_default_graph()
    words_with_sentiments, word_indices, embedding_matrices = load_files()

    labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    input_data = tf.placeholder(tf.int32, [batch_size, max_tweet_words])

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
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    train(word_indices, optimizer, input_data, labels, loss, accuracy, prediction, embedding_matrices)


def train(word_indices, optimizer, input_data, labels, loss, accuracy, prediction, embedding_matrices):
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        # Next Batch of reviews
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
            save_path = saver.save(sess, "data/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)

    inputText = "That movie was awesome good great nice fantastic"
    inputMatrix = getSentenceMatrix(inputText,embedding_matrices)

    sess.run(tf.global_variables_initializer())
    predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
    print("first line:", predictedSentiment)
    if (predictedSentiment[0] > predictedSentiment[1]):
        print("Positive Sentiment")
    else:
        print("Negative Sentiment")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence,embedding_matrices):
    arr = np.zeros([batch_size, max_tweet_words])
    sentenceMatrix = np.zeros([batch_size,max_tweet_words], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = embedding_matrices[4][word]
        except:
            sentenceMatrix[0,indexCounter] = 2 #Vector for unkown words
    return sentenceMatrix

if __name__ == '__main__':
    build_rnn_model()
