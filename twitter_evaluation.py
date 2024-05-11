import tweepy
from datetime import datetime
import config

#Use you keys here
auth = tweepy.OAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
auth.set_access_token(config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

def get_tweets_by_month(username, month):
    current_year = datetime.now().year
    start_date = datetime(current_year, month, 1)
    if month == 12:
        end_date = datetime(current_year + 1, 1, 1)
    else:
        end_date = datetime(current_year, month + 1, 1)

    tweets = []
    tmp_tweets = api.user_timeline(screen_name=username, count=200, tweet_mode='extended')

    while tmp_tweets:
        for tweet in tmp_tweets:
            tweet_date = tweet.created_at
            if start_date <= tweet_date < end_date:
                tweets.append(tweet)
            elif tweet_date < start_date:
                return tweets

        tmp_tweets = api.user_timeline(screen_name=username, count=200, tweet_mode='extended',
                                       max_id=tmp_tweets[-1].id - 1)

    return tweets

def evaluate_tweets_of_month(tweets_results):

    stats = {"real": 0, "fake": 0, "total": 0}

    for tweet in tweets_results:
        stats["total"] += 1
        if tweet == False:
            stats["fake"] += 1
        else:
            stats["real"] += 1

    return stats
