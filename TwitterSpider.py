import json
import pandas as pd
from tweepy.auth import OAuthHandler
from tweepy.api import API
from tweepy.cursor import Cursor
from tweepy import TweepError
import os
import time
import datetime


def OAuth(config):
    """
    Twitter OAuth
    :param config: configuration dictionary
    :return: twitter api instance
    """
    consumer_key = config['twitter_consumer_key']
    consumer_secret = config['twitter_consumer_secret']
    access_token = config['twitter_access_token']
    access_token_secret = config['twitter_access_token_secret']
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return API(auth)


def get_tweets(api, screen_name, since, until, max_pages=200):
    """
    get user's timeline by api

    :param api: twitter api instance
    :param screen_name: user's name shown in TWitter
    :param since: since date, formatted in 'YYYY-MM-DD'
    :param until: until date, formatted in 'YYYY-MM-DD'
    :return: pandas dataframe
    """
    cursor = Cursor(api.user_timeline, screen_name=screen_name, tweet_mode='extended',
                    result_type='recent', count=100,
                    since=since,
                    until=until)

    timeline = []
    try:
        # read by pages
        for tweets in cursor.pages(max_pages):
            timeline.extend(tweet_filter(tweets))
            time.sleep(1)
    except TweepError as e:
        print('A Tweepy error occurred')  # , error code is {}'.format(e.args[0][0]['code']))
        # print(e.args[0][0]['message'])
    return pd.DataFrame(timeline)


def tweet_filter(statuses):
    """
    choose tweets without media and not a retweet.
    :param statuses: tweets statuses
    :return: tweet's id and text in a dictionary
    """
    for tweet_status in statuses:
        tweet_status = tweet_status._json
        if 'media' in tweet_status['entities'] or tweet_status['full_text'].startswith('RT @'):
            continue
        yield {'id': tweet_status['id'], 'text': tweet_status['full_text']}


def get_comments(api, tweets_data, username, since, max_size=5000, duration=1):
    """
    search comments on specific tweets by date.

    :param api: twitter api instance
    :param tweets_data: tweets data with id numbers
    :param username: the target user of search
    :param since: the date start search, format in 'YYYY-MM-DD'
    :param max_size: maximum size of one day search
    :param duration: length of days to search
    :return: pandas dataframe
    """
    since_datetime = datetime.datetime.strptime(since, '%Y-%m-%d')
    ids = tweets_data.loc[:, 'id']
    id_set = set(ids)
    result = []
    for _ in range(0, duration):
        temp_result = []
        until_datetime = since_datetime + datetime.timedelta(days=1)
        # search
        cursor = Cursor(api.search, q='to:{}'.format(username),
                        tweet_mode='extended',
                        count=100,
                        result_type='recent',
                        lang='en',
                        since='{}-{}-{}'.format(since_datetime.year, since_datetime.month, since_datetime.day),
                        until='{}-{}-{}'.format(until_datetime.year, until_datetime.month, until_datetime.day))
        try:
            for comment in cursor.items():
                comment = comment._json
                if 'media' in comment['entities'] or comment['in_reply_to_status_id'] not in id_set or comment[
                    'full_text'].startswith('RT @'):
                    continue
                temp_result.append({'id': comment['id'],
                                    'reply_id': comment['in_reply_to_status_id'],
                                    'user_id': comment['user']['id'],
                                    'text': comment['full_text']})
                if len(temp_result) >= max_size:
                    # when size of comments in a day is greater max_size, stop search
                    break
                time.sleep(0.05)
        except TweepError as e:
            print('A Tweepy error occurred')  # , error code is {}'.format(e.args[0][0]['code']))
            # print(e.args[0][0]['message'])
        result.extend(temp_result)
        since_datetime = until_datetime
        # sleep 3 minutes after finish one search
        time.sleep(60 * 3)
    return pd.DataFrame(result)


with open(os.path.abspath(os.path.dirname(__file__)) + '/config.json', 'r') as config_json:
    config = json.load(config_json)

api = OAuth(config)
# tweets_data = get_tweets(api, screen_name='realDonaldTrump', since='2018-11-18', until='2018-11-24')
# tweets_data.to_csv('./data/Tweets.csv', index=False)

tweets_data = pd.read_csv('./data/Tweets.csv')
comments_data = get_comments(api, tweets_data, 'realDonaldTrump', '2018-11-18', duration=9)
comments_data.to_csv('./data/Comments.csv', header=True, index=False)
