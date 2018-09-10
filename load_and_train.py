import pandas as pd
import re
import os

# TODO using textgenrnn with pretrained model for proof of concept; may or may not do other things later
from textgenrnn import textgenrnn


def replace_junk(tweet):
    """
    Clean data... trying to get meaninful words/sentences/phrases so avoiding links, handles, and extra whitespace
    """
    tweet = tweet.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('"', '')
    tweet = re.sub(r'http[^\s]+', '', tweet)
    tweet = re.sub(r'@[^\s]+', '', tweet)
    return re.sub(r'\s+', ' ', tweet)


def should_use(filename, start, end):
    """
    Check if filename starts and ends how it should
    """
    return filename.startswith(start) and filename.endswith(end)


if __name__ == '__main__':
    TWEET_DIR = 'tweets'
    FILE_START = 'IRAhandle_tweets_'
    MIN_LEN = 10

    csvs = ['{}/{}'.format(TWEET_DIR, f) for f in sorted(os.listdir(TWEET_DIR)) if should_use(f, FILE_START, '.csv')]
    print('Number of files to use: {}'.format(len(csvs)))

    all_tweets = []
    for csv in csvs:
        tweets = pd.read_csv(csv, usecols=['content', 'language'])
        tweets = tweets[(tweets['language'] == 'English')]['content'].dropna()
        cleaned_tweets = [replace_junk(tweet) for tweet in tweets]
        print('File Name: {}, Processed Tweet Quantity: {}'.format(csv, len(cleaned_tweets)))
        all_tweets.extend(cleaned_tweets)

    # don't care about order
    all_tweets_no_dupes = list(set(all_tweets))
    print('Total number of tweets with no post-clean duplicates: {}'.format(len(all_tweets_no_dupes)))

    # get rid of tweets with remaining text under a minimum length
    all_tweets_no_dupes = [t for t in all_tweets_no_dupes if len(t) > MIN_LEN]
    print('Total number of remaining tweets with length > {}: {}'.format(MIN_LEN, len(all_tweets_no_dupes)))

    textgen = textgenrnn(name='538_russian_tweets')
    textgen.train_on_texts(all_tweets_no_dupes, num_epochs=10, gen_epochs=1, train_size=1.0, batch_size=1024)

