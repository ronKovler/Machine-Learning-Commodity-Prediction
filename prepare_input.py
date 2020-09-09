#!/usr/bin/env python
##############################################################
#
#   Prepare inputs for Keras training
#
#   Alexey Goder
#   agoder@yahoo.com
#
#   august 29th, 2019
#
##############################################################
import csv
import pickle
import numpy
import calendar
from datetime import datetime, date, timedelta
from production import get_stop_words, clean_and_stem, classify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from classification import clean_and_stem


ANALYSIS_WINDOW_SIZE = 7
FORECAST_RANGE_IN_WEEKS = 10

def get_tweets(file, include_nick=False, range='weekly', classifier=None, stop_words=None, size=19):
    output = {}

    with open(file, 'rt', encoding="utf8") as tweetCsv:
        rowReader = csv.reader(tweetCsv, delimiter=',')
        next(rowReader)
        for row in rowReader:
            tweet_text = row[3].replace('\n', '').split('http')[0]

            if classifier is not None and not classify(tweet_text, classifier, stop_words=stop_words)[0]:
                continue

            if include_nick:
                tweet_text += ' ' + row[1]
            tweet_index = get_tweet_index_from_date(datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S').date(), range=range)
            output[tweet_index] = output.setdefault(tweet_index, '') + (' ' + tweet_text)

            if len(output) > size:
                break
    return output


def get_tweets_from_pickle(commodity, file_path='filtered_tweets_'):
    with open(file_path + commodity + '.pkl', 'rb') as f:
        tweet_dict = pickle.load(f)
    return tweet_dict


def get_spot_prices(file):
    spot = {}
    with open(file, 'rt', encoding="utf8") as tweetCsv:
        rowReader = csv.reader(tweetCsv, delimiter=',')
        for row in rowReader:
            date_list = [int(x) for x in row[0].split('/')]
            d = date(date_list[2], date_list[0], date_list[1])
            spot[d] = float(row[1])
    return spot

def get_tweet_index_from_date(tweet_date, range='weekly'):
    if range == 'weekly':
        return tweet_date.isocalendar()[0]*100 + tweet_date.isocalendar()[1]
    elif range == 'biweekly':
        tweet_index = tweet_date.isocalendar()[0] * 100 + tweet_date.isocalendar()[1]
        return tweet_index + (tweet_index%2)
    else:
        return None


def get_spot_price_by_date(d, spot):
    start_date = d
    for i in range(5):
        if start_date in spot:
            return spot[start_date]
        start_date += timedelta(days=1)
    return None


def get_dates_from_tweet_index(tweet_index, range='weekly'):

    if tweet_index is None:
        return None, None

    tweet_year = int(tweet_index/100)
    tweet_week = tweet_index%100

    if range == 'weekly':
        firstday = datetime.strptime(f'{tweet_year}-W{int(tweet_week) - 1}-1', "%Y-W%W-%w").date()
        lastday = firstday + timedelta(days=6.9)
    elif range == 'biweekly':
        firstday = datetime.strptime(f'{tweet_year}-W{int(tweet_week - 1) - 1}-1', "%Y-W%W-%w").date()
        lastday = firstday + timedelta(days=13.9)
    else:
        return None

    return firstday, lastday


def get_last_friday():
    lastFriday =  date.today()
    oneday = timedelta(days=1)

    while lastFriday.weekday() != calendar.FRIDAY:
        lastFriday -= oneday
    return lastFriday


def get_monday_after_last_friday():
    return get_last_friday() + timedelta(days=3)


def advance_to_next_workday(day):
    day += timedelta(days=1)
    while day.weekday() == calendar.SATURDAY or day.weekday() == calendar.SUNDAY:
        day += timedelta(days=1)
    return day


def tweets_to_vectors(texts, stop_words=set(), max_f=100, idf=True, ngram=(1, 5), counter=None, transformer=None):

    if counter is None or transformer is None:
        counter = CountVectorizer(ngram_range=ngram,
                                     analyzer='word',
                                     max_df=1.0,
                                     min_df=0,
                                     max_features=max_f,
                                     )
        transformer = TfidfTransformer(use_idf=idf)
        counts = counter.fit_transform([clean_and_stem(t, stop_words) for t in texts])
        vec = transformer.fit_transform(counts)
    else:
        counts = counter.transform([clean_and_stem(t, stop_words) for t in texts])
        vec = transformer.transform(counts)

    return vec, counter, transformer


def train_test_split(t_dict):
    all_keys = list(t_dict.keys())
    return all_keys[::2], all_keys[1::2]


def prepare_input(t_keys, vector, price_dict, weeks=FORECAST_RANGE_IN_WEEKS, counter=None):
    x_input = []
    x_keys = []
    y_output = []

    for i, k in enumerate(t_keys):
        f, l = get_dates_from_tweet_index(k)
        spot_base = get_spot_price_by_date(l, price_dict)
        target_list = [get_spot_price_by_date(l+ timedelta(days=(delta + 1)*ANALYSIS_WINDOW_SIZE),price_dict)
                       for delta in range(weeks)
                       ]

        if spot_base is not None and  None not in target_list and spot_base > 0.0:
            relative_spot_list = [100.0*(t - spot_base)/spot_base for t in target_list]
            x_input.append(vector.toarray()[i,].tolist())
            x_keys.append(k)
            y_output.append(relative_spot_list)

    return x_keys, list2array(x_input), list2array(y_output), counter


def get_train_and_test(t_dict, max_f=100, idf=True, ngram=(1, 2)):
    stop_words = get_stop_words()
    train_keys, test_keys = train_test_split(t_dict)
    train_vec, counter, transformer = tweets_to_vectors([t_dict[k] for k in train_keys],
                                                        max_f=max_f,
                                                        idf=idf,
                                                        ngram=ngram,
                                                        stop_words=stop_words
                                                        )
    test_vec, counter, transformer = tweets_to_vectors([t_dict[k] for k in test_keys],
                                                       stop_words=stop_words,
                                                       counter=counter,
                                                       transformer=transformer
                                                       )
    return train_vec, test_vec, counter


def get_data(commodity, train=True, max_f=100, idf=True, ngram=(1, 2)):
    tweet_dict = get_tweets_from_pickle(commodity)
    train_keys, test_keys = train_test_split(tweet_dict)
    train_vec, test_vec, counter = get_train_and_test(tweet_dict, max_f=max_f, idf=idf, ngram=ngram,)
    if train:
        return train_keys, train_vec, counter
    else:
        return test_keys, test_vec, counter


def list2array(l):
    return numpy.array([numpy.array(xi) for xi in l])


if __name__ == "__main__":

    spot_price = get_spot_prices('cattle_spot_prices.csv')
    COMMODITY = 'cattle'

    print(get_last_friday(), get_monday_after_last_friday())
    print(advance_to_next_workday(get_last_friday()), advance_to_next_workday(get_monday_after_last_friday()))

    train_keys, train_vec, counter = get_data(COMMODITY)
    print(train_vec.shape)

    x_keys, x_input, y_output, counter = prepare_input(train_keys, train_vec, spot_price, weeks=FORECAST_RANGE_IN_WEEKS)
    print(x_keys)
    print(len(y_output), len(y_output[0]))
    print(len(x_keys))
