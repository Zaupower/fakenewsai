import tweepy
from datetime import datetime
import pandas as pd
import config

# Use your keys here
print("CONSUMER_KEY: " + config.CONSUMER_KEY)
print("CONSUMER_SECRET: " + config.CONSUMER_SECRET)
print("ACCESS_TOKEN: " + config.ACCESS_TOKEN)
print("ACCESS_TOKEN_SECRET: " + config.ACCESS_TOKEN_SECRET)
#Pass in our twitter API authentication key
auth = tweepy.OAuth1UserHandler(
    config.CONSUMER_KEY, config.CONSUMER_SECRET,
    config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET
)
#Instantiate the tweepy API
api = tweepy.API(auth, wait_on_rate_limit=True)

def get_latest_tweet(username):
    search_query = "'ref''world cup'-filter:retweets AND -filter:replies AND -filter:links"
    no_of_tweets = 100
    try:
        # The number of tweets we want to retrieved from the search
        tweets = api.search_tweets(q=search_query, lang="en", count=no_of_tweets, tweet_mode='extended')

        # Pulling Some attributes from the tweet
        attributes_container = [[tweet.user.name, tweet.created_at, tweet.favorite_count, tweet.source, tweet.full_text]
                                for tweet in tweets]

        # Creation of column list to rename the columns in the dataframe
        columns = ["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"]

        # Creation of Dataframe
        tweets_df = pd.DataFrame(attributes_container, columns=columns)
    except BaseException as e:
        print('Status Failed On,', str(e))

def get_tweets_by_month(username, month):
    current_year = datetime.now().year
    start_date = datetime(current_year, month, 1)
    if month == 12:
        end_date = datetime(current_year + 1, 1, 1)
    else:
        end_date = datetime(current_year, month + 1, 1)

    tweets = []

    try:
        tmp_tweets = api.user_timeline(screen_name=username, count=200, tweet_mode='extended')
    except tweepy.errors.Unauthorized as e:
        print(f"Unauthorized error: {e}")
        return {"error": "Unauthorized access - Invalid or expired token"}
    except tweepy.errors.TooManyRequests as e:
        print(f"Rate limit exceeded: {e}")
        return {"error": "Rate limit exceeded, please try again later"}
    except tweepy.errors.TweepyException as e:
        print(f"Error fetching tweets: {e}")
        return {"error": str(e)}

    while tmp_tweets:
        for tweet in tmp_tweets:
            tweet_date = tweet.created_at
            if start_date <= tweet_date < end_date:
                tweets.append(tweet.full_text)
            elif tweet_date < start_date:
                return tweets

        try:
            tmp_tweets = api.user_timeline(screen_name=username, count=200, tweet_mode='extended',
                                           max_id=tmp_tweets[-1].id - 1)
        except tweepy.errors.Unauthorized as e:
            print(f"Unauthorized error: {e}")
            return {"error": "Unauthorized access - Invalid or expired token"}
        except tweepy.errors.TooManyRequests as e:
            print(f"Rate limit exceeded: {e}")
            return {"error": "Rate limit exceeded, please try again later"}
        except tweepy.errors.TweepyException as e:
            print(f"Error fetching tweets: {e}")
            return {"error": str(e)}

    return tweets

def evaluate_tweets_of_month(tweets_results):
    if "error" in tweets_results:
        return tweets_results

    stats = {"real": 0, "fake": 0, "total": 0}

    for tweet in tweets_results:
        stats["total"] += 1
        if tweet == False:
            stats["fake"] += 1
        else:
            stats["real"] += 1

    return stats
