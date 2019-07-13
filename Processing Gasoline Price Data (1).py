#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from decimal import Decimal
from datetime import datetime
import matplotlib.dates as dates
from matplotlib import pyplot as plt

with open('GasolineFutures-HistoricalData.csv', 'rt') as csvfile:
    rowreader = csv.reader(csvfile, delimiter=',')
    count = 0
    date_array = []
    delta_array = []
    for row in rowreader:
        if(count != 0):
            date_array.insert(0, datetime.strptime(row[0], '%b %d, %Y'))
            delta_array.insert(0, Decimal(row[6].strip('%')))
        count+=1
        

week_array = []
weekly_delta_array = []
length = len(date_array)
sumVal = 0
for i in range(length):
    if(i < length-1):
        sumVal += delta_array[i]
        if(date_array[i].weekday() == 4):
            #print("worked")
            weekly_delta_array.append(sumVal)
            sumVal = 0
            week_array.append(date_array[i])
        


# In[2]:


plt.plot(date_array, delta_array)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Change in %', fontsize=16)


# In[3]:


plt.plot(week_array, weekly_delta_array)
plt.xlabel('Date - Weekly Information Summed to Every Friday', fontsize=18)
plt.ylabel('Change in %', fontsize=16)


# In[4]:


#date_time = now.strftime("%m/%d/%Y")
#print("date and time:",date_time)

top_row = ['Week of', 'Weekly Change in %']
output = []
output.append(top_row)
for i in range(len(week_array)):
    row = [week_array[i].strftime("%m/%d/%Y"), weekly_delta_array[i]]
    output.append(row)
    print(output[i])


# In[6]:


out


# In[ ]:




