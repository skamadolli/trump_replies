import string
import re
import pandas as pd

URL_REGEX = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
MENTION_REGEX = r"@\w{1,15}"
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)


def process_data():
    target_with_sentences = read_csv()
    for i in range(len(target_with_sentences)):
        words = str(target_with_sentences[i][1]).split()
        target_with_sentences[i][1] = clean_up_words(words)
    return target_with_sentences


def read_csv():
    df = pd.read_csv('tweets_with_sentiment.zip',
                     encoding='ISO-8859-1',
                     header=None,
                     names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    df = df.drop(columns=['ids', 'date', 'flag', 'user'])
    return df.head(100).to_numpy()


def clean_up_words(words):
    internet_free = [word for word in words if
                     (re.match(pattern=URL_REGEX, string=word) is None) and
                     (re.match(pattern=MENTION_REGEX, string=word) is None)]
    case_free = [word.lower() for word in internet_free]
    emoji_free = [word for word in case_free if not word.startswith(':') and not word.startswith(';')]
    punctuation_free = [word.translate(PUNCTUATION_TABLE) for word in emoji_free]
    empty_string_free = [word for word in punctuation_free if word]
    return empty_string_free


if __name__ == '__main__':
    process_data()
