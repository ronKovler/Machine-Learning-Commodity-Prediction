#!/usr/bin/env python
##############################################################
#
#   Optimize classifier parameters for document relevance
#
#   Alexey Goder
#   agoder@yahoo.com
#
#   july 24th, 2019
#
##############################################################

import string
import re
import pickle
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

COMMODITY = 'oil'
positive_file = 'C:/Users/agode/Documents/AGBlox/ai/train_' + COMMODITY + '_twits.txt'
negative_file = 'C:/Users/agode/Documents/AGBlox/ai/train_not_' + COMMODITY + '_twits.txt'
positive_test = 'C:/Users/agode/Documents/AGBlox/ai/test_' + COMMODITY + '_twits.txt'
negative_test = 'C:/Users/agode/Documents/AGBlox/ai/test_not_' + COMMODITY + '_twits.txt'


def clean_and_stem(text, stop_words=[]):
    """
    Remove punctuation and stem
    :param text: string
    :param stop_words:set of stop words
    :return: string
    """
    stemmer = SnowballStemmer('english')
    tmp = text.replace('\n', ' ').replace('”', ' ').replace('“', ' ').replace('…', ' ')
    return ' '.join(
        [x for x in [stemmer.stem(w) for w in re.sub('[' + string.punctuation + ']', ' ', tmp).split(' ')] if
         len(x) and x not in stop_words])


def get_tweets(file, raw=False, stop_words=set()):
    """
    Load twits from a file
    :param file: string
    :param raw: True if no cleaning, false otherwise
    :param stop_words: set of stop words
    :return: list of string
    """
    with open(file, encoding="utf8") as f:
        if raw:
            return [tweet.replace('\n', '').split('http')[0] for tweet in f]
        else:
            return [clean_and_stem(tweet.replace('\n', '').split('http')[0], stop_words) for tweet in f]


def get_category_name(file_name):
    """
    Convert file name into a commodity name
    :param file_name: string
    :return: string
    """
    return file_name.split('/')[-1].replace('.txt', '').replace('train_', '').replace('test_', '')


def train(positive_file,
          negative_file,
          stop_words=set(),
          max_f=100,
          size=2000,
          idf=True,
          ngram=(1, 2),
          alpha=1.0,
          classifier=MultinomialNB(alpha=0.15)
          ):
    """
    Train a classifier
    :param positive_file:
    :param negative_file:
    :param stop_words:
    :param max_f:
    :param size:
    :param idf:
    :param ngram:
    :param alpha:
    :param classifier:
    :return:
    """
    category = get_tweets(positive_file, stop_words=stop_words)[:size]
    not_category = get_tweets(negative_file, stop_words=stop_words)[:size]
    target = [1] * len(category) + [2] * len(not_category)

    count_vect = CountVectorizer(ngram_range=ngram,
                                 analyzer='word',
                                 max_df=1.0,
                                 min_df=0,
                                 max_features=max_f,
                                 )

    X_train_counts = count_vect.fit_transform(category + not_category)
    tfidf_transformer = TfidfTransformer(use_idf=idf)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return classifier.fit(X_train_tfidf, target), count_vect, tfidf_transformer


def test(clf, count_vect, tfidf_transformer):
    """
    Test a classifier
    :param clf:
    :param count_vect:
    :param tfidf_transformer:
    :return:
    """
    category_test = get_tweets(positive_test)
    not_category_test = get_tweets(negative_test)

    docs_new = category_test + not_category_test
    target = [1] * len(category_test) + [2] * len(not_category_test)
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    return clf.score(X_new_tfidf, target)


if __name__ == "__main__":

    # Load stop words
    stop_words = set()
    with open('C:/Users/agode/Documents/AGBlox/stop_words.txt') as f:
        for word in f:
            w = word.replace('/n', '').strip()
            if len(w):
                stop_words.add(word.replace('/n', '').strip())

    # Define input files

    target_names = {1: get_category_name(positive_file), 2: get_category_name(negative_file)}

    best = -1
    # BEST: cattle Size=1800 Max_f=2500 IDF=False ngram= (1, 4) alpha= 0.2 AdaBoost 0.9803314023979524
    # BEST: corn Size=500 Max_f=100 IDF=False ngram= (1, 2) Neural Net 0.9956890744981813
    # BEST: oil Size=41 Max_f=10 IDF=False ngram= (1, 1) Decision Tree 0.9997305671561363
    # BEST: diesel Size=12 Max_f=10 IDF=False ngram= (1, 1) Decision Tree 1.0
    for idf in [False, True]:
        for max_f in range(10, 100, 10):
            for size in range(41, 42, 1):
                for ngram in [(1, 1), (1, 2), (1, 3)]:

                    classifier_names = [
                        # "Naive Bayes",
                        "Nearest Neighbors",
                        "Linear SVM",
                        "Decision Tree",
                        # "Random Forest",
                        "Neural Net",
                        "AdaBoost"
                    ]
                    classifiers = [
                        # MultinomialNB(alpha=0.15),
                        KNeighborsClassifier(3),
                        SVC(kernel="linear", C=0.025),
                        DecisionTreeClassifier(max_depth=5),
                        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                        MLPClassifier(alpha=1, max_iter=1000),
                        AdaBoostClassifier()]

                    for i, classifier in enumerate(classifiers):
                        alpha = 0.05 * i
                        clf, count_vect, tfidf_transformer = train(positive_file,
                                                                   negative_file,
                                                                   stop_words=stop_words,
                                                                   max_f=max_f,
                                                                   size=size,
                                                                   idf=idf,
                                                                   ngram=ngram,
                                                                   alpha=alpha,
                                                                   classifier=classifier
                                                                   )
                        res = test(clf, count_vect, tfidf_transformer)
                        print('Size=%s' % size, 'Max_f=%s' % max_f, 'IDF=%s' % idf, 'ngram=', ngram,
                              classifier_names[i], res)

                        if res > best:
                            best = res
                            best_size = size
                            best_idf = idf
                            best_max_f = max_f
                            best_ngram = ngram
                            best_alpha = alpha
                            best_class = i
                            category_classifier = {'clf': clf, 'trnsf': tfidf_transformer, 'cnt': count_vect}
                            with open('category_' + COMMODITY + '.pkl', 'wb') as f:
                                pickle.dump(category_classifier, f)
                            print('BEST:', COMMODITY, 'Size=%s' % best_size, 'Max_f=%s' % best_max_f,
                                  'IDF=%s' % best_idf, 'ngram=', ngram, classifier_names[best_class], best)
