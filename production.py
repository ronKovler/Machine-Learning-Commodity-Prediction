#!/usr/bin/env python
##############################################################
#
#   Classifier for production
#
#   Alexey Goder
#   agoder@yahoo.com
#
#   july 31st, 2019
#
##############################################################

import string
import re
import pickle
from nltk.stem.snowball import SnowballStemmer


CLASSIFIER_FOLDER = ''
STOP_WORDS_FILE = 'stop_words.txt'

def load_classifier(classifier_name):
    """
    Load a classifier from a pickle file
    :param classifier_name: string
    :return: dictionary with the classifier structure
    """

    try:
        with open(CLASSIFIER_FOLDER + 'category_' + classifier_name + '.pkl', 'rb') as f:
            classifier =  pickle.load(f)
        return classifier
    except:
        return None


def classify(text, category_classifier, stop_words):
    """
    Classify a text
    :param text: string
    :param category_classifier: dictionary with the classifier structure
    :param stop_words: set
    :return: tupple: True or False and a probality between 0 and 1
    """
    cleaned_text = clean_and_stem(text, stop_words = stop_words)
    vec = category_classifier['trnsf'].transform(category_classifier['cnt'].transform([cleaned_text]))
    return category_classifier['clf'].predict(vec)[0] == 1, category_classifier['clf'].predict_proba(vec)[0][0]


def get_stop_words(file_name=STOP_WORDS_FILE):
    """
    Load stopwords
    :param file_name: string
    :return: set of stop words
    """
    stop_words = set()
    with open(file_name) as f:
        for word in f:
            w = word.replace('/n', '').strip()
            if len(w):
                stop_words.add(word.replace('/n', '').strip())
    return stop_words


def clean_and_stem(text, stop_words=[]):
    """
    Remove punctuation and stem
    :param text: string
    :param stop_words:set of stop words
    :return: string
    """
    stemmer = SnowballStemmer('english')
    tmp = text.replace('\n', ' ').replace('”', ' ').replace('“', ' ').replace('…', ' ')
    return ' '.join([x for x in [stemmer.stem(w) for w in re.sub('['+string.punctuation+']', ' ', tmp).split(' ')] if len(x) and x not in stop_words])


if __name__ == "__main__":

    # Load classifier
    category_classifier = load_classifier('oil')

    # Load stop words
    stop_words = get_stop_words()

    # Sample use
    twit = '🇺🇸🤝🇧🇷Investors are welcome. Those who arrive first will have the best opportunities or as we say here in an old s…  StaLuziaEsteio @MCastelloes @DDFalpha Obrigado. StaLuziaEsteio Brazil has 32,140 new job'
    result = classify(twit, category_classifier, stop_words=stop_words)
    print(result)
