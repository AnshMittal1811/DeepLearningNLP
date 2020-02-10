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