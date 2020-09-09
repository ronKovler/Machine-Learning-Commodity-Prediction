#!/usr/bin/env python
##############################################################
#
#   Get input data for proce forecast
#
#   Alexey Goder
#   agoder@yahoo.com
#
#   august 29th, 2019
#
##############################################################

import json
import requests
import calendar
from datetime import datetime, date, timedelta
from typing import Iterator
from requests.auth import HTTPBasicAuth
from production import get_stop_words, load_classifier, classify, clean_and_stem
from prepare_input import get_last_friday
from sklearn.feature_extraction.text import TfidfTransformer


PROD = "https://lab.agblox.com/api"
DEV = "http://localhost:5000/api"
TWEET_END = "/tweets"
ENDPOINT = "/predictions"
TWEET_ENDPOINT = "/tweets"
COMMODITY_ENDPOINT = "/commodities"
USER = "admin"
PASS = "zt585nEiE4SpmGg4qv9Wr6bFy"

COMMODITY = 'cattle'
MIN_KEYWORD_COUNT = 500

headers = {'content-type' : 'application/json'}
category_classifier = load_classifier(COMMODITY)
stop_words = get_stop_words()


def get_tweets_from_api(start_date="2019-08-27"):
    params = {"start": start_date}
    r = requests.get(PROD + TWEET_ENDPOINT, auth=HTTPBasicAuth(USER, PASS), params=params)
    return json.loads(r.text)


def get_commodity_price_from_api(product_code="GF", expiration_code="U9", start_date="2019-08-25"):
    params = {"start": start_date, "product_code": product_code, "expiration_code": expiration_code}
    r = requests.get(PROD + COMMODITY_ENDPOINT, auth=HTTPBasicAuth(USER, PASS), params=params)
    return json.loads(r.text)


def get_last_price(price_list):
    if len(price_list):
        return sorted(price_list, reverse=True, key=lambda e: e['created_at'])[0]['last']
    else:
        return 140.0


def get_last_commodity_price(product_code="GF", expiration_code="U9"):
    start_date = (get_last_friday() - timedelta(days=5))
    return get_last_price(get_commodity_price_from_api(product_code=product_code,
                                                       expiration_code=expiration_code,
                                                       start_date=start_date.strftime("%m/%d/%Y")
                                                       )
                          )


def get(url: str = DEV, endpoint: str = ENDPOINT, user: str = USER, password: str = PASS) -> Iterator[dict]:
    r = requests.get(f"{url}{endpoint}", auth=HTTPBasicAuth(user, password))
    for each in r.json():
        yield each


def is_tweet_relevant(text, stop_words=stop_words):
    clean = text.replace('\n', '').split('http')[0]
    if classify(clean, category_classifier, stop_words=stop_words)[0]:
        return clean
    else:
        return ''


def get_recent_tweets(stop_words=stop_words):
    start_date = (get_last_friday() - timedelta(days=7))
    keywords = []
    while len(keywords) < MIN_KEYWORD_COUNT:
        raw = get_tweets_from_api(start_date=start_date.strftime("%m/%d/%Y"))

        if 'tweets' in raw:
            list_of_tweets = raw['tweets']
        else:
            list_of_tweets = []

        relevant_list = [is_tweet_relevant(t['text'].replace('\n', '').split('http')[0], stop_words=stop_words)
                         for t in list_of_tweets
                         ]
        keywords = ' '.join(relevant_list).split()
        start_date -= timedelta(days=5)

    return ' '.join(relevant_list)


def text_to_vector(text, stop_words=stop_words, idf=True, counter=None):
    counts = counter.fit_transform([clean_and_stem(text, stop_words)])
    return TfidfTransformer(use_idf=idf).fit_transform(counts)


def get_input_vector(counter=None):
    return text_to_vector(get_recent_tweets(), idf=True, counter=counter)


if __name__ == '__main__':

    print(get_last_commodity_price())
