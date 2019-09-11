from processors import words
import pickle
import numpy as np

# Process tweet data
target_with_sentences = words.process_data()
num_of_tweets = len(target_with_sentences)
max_tweet_words = 20

# Import Embedding
embed = {}
with open('../data/embeddings.pkl', 'rb') as handle:
    # noinspection PyRedeclaration
    embed = pickle.load(handle)

# Create ID's Matrix
ids = np.zeros((num_of_tweets, max_tweet_words), dtype='int32')
tweet_counter = 0
for tweet in target_with_sentences:
    word_counter = 0
    for word in tweet[1]:
        if word_counter < max_tweet_words:
            try:
                ids[tweet_counter][word_counter] = embed[4][word]
            except KeyError:
                ids[tweet_counter][word_counter] = 10001
        word_counter += 1
    # print(tweet[1])
    # print(ids[tweet_counter])
    tweet_counter += 1

np.save('data/tweet_words_ids_matrix.npy', ids)
