#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Load stop words
stop_words = set()
#C:/Users/Ron/Documents/DEEEP LEARNING COMMODITY PRED/stop_words.txt
with open('C:/Users/Ron/Documents/DEEEP LEARNING COMMODITY PRED/stop_words.txt') as f:
    for word in f:
        w = word.replace('/n', '').strip()
        if len(w):
            stop_words.add(word.replace('/n', '').strip())


# In[5]:


# Remove punctuation and stem

from nltk.stem.snowball import SnowballStemmer

def clean_and_stem(text, stop_words = []):
    stemmer = SnowballStemmer('english')
    tmp = text.replace('\n', ' ').replace('”', ' ').replace('“', ' ').replace('…', ' ')
    return ' '.join([x for x in [stemmer.stem(w) for w in re.sub('['+string.punctuation+']', ' ', tmp).split(' ')] if len(x) and x not in stop_words])


# In[28]:


import csv
from datetime import datetime
import pixiedust

def get_tweets(file, raw = False, stop_words = stop_words, time_interval = 'weekly', include_nick = False):
    with open(file, 'rt', encoding="utf8") as tweetCsv:
        rowReader = csv.reader(tweetCsv, delimiter=',')
        count = 0
        tweetIntervalString = ''
        tweet_array = []
        date_array = []
        week_array = []
        for row in rowReader:
            if(count == 0):
                #Skips first row of csv file containing column headers
                count+=1
                continue
            tweetText = row[3].replace('\n', '').split('http')[0]
            if(include_nick):
                tweetText += ' ' + row[1]
            tweet_array.insert(0, tweetText)   
            date_array.insert(0, datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S'))
            count+=1
            #print(row[3].replace('\n', '').split('http')[0])
        
        startDate = date_array[0]
        currentDate = None
        output = []
        
        for i in range(len(date_array)):
            currentDate = date_array[i]
            if(within_time_interval(time_interval, startDate, currentDate)):
                if(raw):
                    tweetIntervalString += ' '+tweet_array[i]
                else:
                    tweetIntervalString += ' '+clean_and_stem(tweet_array[i], stop_words)
            else:
                output.append(tweetIntervalString)
                tweetIntervalString = tweet_array[i]
                startDate = date_array[i]   #CHANGE MADE HERE <-----------------------------
   
        return output
    
def within_time_interval(interval, startDate, currentDate):
    if(interval == 'weekly'):
        print(currentDate.toordinal() - startDate.toordinal())
        if(currentDate.weekday() == 6 and currentDate.toordinal() - startDate.toordinal() >= 7) or (startDate.weekday() != 6 and currentDate.weekday() == 6):
            return False
        else:
            return True
    if(interval == 'biweekly'):
        print(currentDate.toordinal() - startDate.toordinal())
        if(currentDate.weekday() == 6 and currentDate.toordinal() - startDate.toordinal() >= 14) or(startDate.weekday() != 6 and currentDate.weekday() == 6):
            return False
        else:
            return True
    if(interval == 'monthly'):
        if(currentDate.month() != startDate.month()):
            return False
        else:
            return True


# In[29]:



import re
import string

x = get_tweets('C:/Users/Ron/Documents/DEEEP LEARNING COMMODITY PRED/tweetDataTester.csv')


# In[30]:


for i in range(len(x)):
    print(x[i] + '\n')


# In[ ]:


print(x)


# In[ ]:




