#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load stop words
stop_words = set()
#C:/Users/Ron/Documents/DEEEP LEARNING COMMODITY PRED/stop_words.txt
with open('C:/Users/Ron/Documents/DEEEP LEARNING COMMODITY PRED/stop_words.txt') as f:
    for word in f:
        w = word.replace('/n', '').strip()
        if len(w):
            stop_words.add(word.replace('/n', '').strip())


# In[ ]:


# Remove punctuation and stem

from nltk.stem.snowball import SnowballStemmer

def clean_and_stem(text, stop_words = []):
    stemmer = SnowballStemmer('english')
    tmp = text.replace('\n', ' ').replace('”', ' ').replace('“', ' ').replace('…', ' ')
    return ' '.join([x for x in [stemmer.stem(w) for w in re.sub('['+string.punctuation+']', ' ', tmp).split(' ')] if len(x) and x not in stop_words])


# In[ ]:


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
            date_array.insert(0, datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S').date())
            
            count+=1
            
        
        startDate = date_array[0]
        currentDate = None
        tweets = []
        output = {}
        for i in range(len(date_array)):
            currentDate = date_array[i]
            if(within_time_interval(time_interval, startDate, currentDate)):
                if(raw):
                    tweetIntervalString += ' '+tweet_array[i]
                else:
                    tweetIntervalString += ' '+clean_and_stem(tweet_array[i], stop_words)
            else:
                output.update({startDate : tweetIntervalString })
                tweetIntervalString = tweet_array[i]
                startDate = date_array[i]
   
        return output #Dictionary with key:beggining week date. and value:tweets from that week
    
def within_time_interval(interval, startDate, currentDate):
    if(interval == 'weekly'):
        #print(currentDate.strftime("%U"))
        if(currentDate.weekday() == 6 and currentDate.toordinal() - startDate.toordinal() >= 7):
            return False
        elif(startDate.weekday() != 6 and currentDate.weekday() == 6):
            return False
        elif(currentDate.strftime("%U") != startDate.strftime("%U")):
            return False
        else:
            return True
    if(interval == 'biweekly'):
        #print(currentDate.toordinal() - startDate.toordinal())
        if(currentDate.weekday() == 6 and currentDate.toordinal() - startDate.toordinal() >= 14):
            return False
        elif(startDate.weekday() != 6 and currentDate.weekday() == 6):
            return False
        else:
            return True
    if(interval == 'monthly'):
        if(currentDate.month() != startDate.month()):
            return False
        else:
            return True


# In[ ]:



import re
import string
#get_tweeets returns a dictionary of key:week date, value: week tweets
x = get_tweets('C:/Users/Ron/Documents/DEEEP LEARNING COMMODITY PRED/tweetData.csv')


# In[ ]:


for key in x.keys():
    
    print( '\n' + x[key]+ '\n')


# In[ ]:


import re
def remove_numericals(dictionary ,numeric_length):
    tweetDict = dictionary
    cond1=r"(\s+)\d{1,"+str(numeric_length)+r"}(\s+)"
    cond2=r"(\s+)\d{1,"+str(numeric_length)+r"}$"
    cond3=r"^\d{1,"+str(numeric_length)+r"}(\s+)"
    
    for key in tweetDict.keys():
        val1=''
        flag=False
        while val1 != tweetDict[key]:
            if flag:
                tweetDict[key] = val1
            val1=re.sub(cond1,r" ",tweetDict[key])
            val1=re.sub(cond2,r"",val1)
            val1=re.sub(cond3,r"", val1)
            flag=True
        


# In[ ]:


remove_numericals(x, 3)

