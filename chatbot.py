# Building chatbot with D-NLP


# Importing the Libraries
import numpy as np
import tensorflow as tf
import re
import time




############### PART 1 - DATA PREPROCESSING
# Import Data
m_lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
m_conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Creating a Dictionary for mapping lines and id
id2mline = {}
for i in m_lines:
    _i = i.split(' +++$+++ ')
    if len(_i) == 5:
        id2mline[_i[0]] = _i[4]
        
# Create list of all converstations
conv_ids = []
for j in m_conversations[:-1]:
    _j = j.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conv_ids.append(_j.split(','))

# Separately Qustions and Answers
que = []
ans = []

for k in conv_ids: 
    for l in range(len(k) - 1):
        que.append(id2mline[k[l]])
        ans.append(id2mline[k[l+1]])
        
# Doing first cleaning of texts
