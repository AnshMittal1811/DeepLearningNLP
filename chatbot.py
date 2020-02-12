# Building chatbot with D-NLP
# Importing the Libraries
import numpy as np
import tensorflow as tf
import re
import time

############### PART 1 - DATA PREPROCESSING ##################
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
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Cleaning all questions
cque = []
for i in que:
    cque.append(clean_text(i))

# Cleaning all answers
cans = []
for j in ans:
    cans.append(clean_text(j))
    
# Creating a Dictionary that map each word to number of occurences
word2count = {}

for question in cque:
    for word in question.split():
        if word not in word2count: 
            word2count[word] = 1
        else:
            word2count[word] +=1
            
for answer in cans:
    for word in answer.split():
        if word not in word2count: 
            word2count[word] = 1
        else:
            word2count[word] +=1
            
# Tokenization and filtering non frequent works
thresh = 20
queswords2int = {}
w_number = 0

for word, count in word2count.items():
    if count >= thresh:
        queswords2int[word] = w_number
        w_number += 1

answords2int = {}
w_number = 0

for word, count in word2count.items():
    if count >= thresh:
        answords2int[word] = w_number
        w_number += 1
        
# Adding the last tokens to 2 dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens: 
    queswords2int[token] = len(queswords2int) + 1
    
for token in tokens: 
    answords2int[token] = len(answords2int) + 1

# Inverse dictionary Mapping of the answords2int
ansints2word = {w_i: w for w, w_i in answords2int.items()}

# Adding the EOS token at end of every answer
for i in range(len(cans)): 
    cans[i] += ' <EOS>'
    
# Translating all ques and answers into integers 
#& Replace all the words filtered out by <OUT>
question2int = []
for questions in cque:
    ints = []
    for word in questions.split():
        if word not in queswords2int: 
            ints.append(queswords2int['<OUT>'])
        else:
            ints.append(queswords2int[word])
    question2int.append(ints)

answers2int = []
for answers in cans:
    ints = []
    for word in answers.split():
        if word not in answords2int: 
            ints.append(answords2int['<OUT>'])
        else:
            ints.append(answords2int[word])
    answers2int.append(ints)
    
# Sort questions and answers by the length of questions
sorted_cque = []
sorted_cans = []

for length in range(1, 26): 
    for l in enumerate(question2int):
        if len(l[1]) == length:
            sorted_cque.append(question2int[l[0]])
            sorted_cans.append(answers2int[l[0]])



############ PART 2 - BUILDING THE SEQ2SEQ MODEL ##############
# Creating placeholders for the answers and the targets
            