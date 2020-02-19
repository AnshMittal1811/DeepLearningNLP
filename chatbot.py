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
# Creating placeholders for the inputs and the targets
def model_inputs(): 
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_probability')
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocesed_targets = tf.concat([left_side, right_side], axis = 1)
    return preprocesed_targets

# Ecoding RNN layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell, 
                                                       sequence_length = sequence_length, 
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state

# Decoding Training set
def decode_training_set(encoder_state, 
                        decoder_cell, 
                        decoder_embedded_input, 
                        sequence_length, 
                        decoding_scope, 
                        output_function, 
                        keep_prob,
                        batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attenton_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                   attention_option='bahdanau',
                                                                                                                                   num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function, 
                                                                              attenton_construct_function,
                                                                              name = 'attn_dec_train')
    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


# Decoding Test/Validation set
def decode_test_set(encoder_state,
                    decoder_cell,
                    decoder_embeddings_matrix,
                    sos_id, eos_id,
                    maximum_length,
                    num_words,
                    decoding_scope,
                    output_function,
                    keep_prob,
                    batch_size):
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attenton_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                   attention_option='bahdanau',
                                                                                                                                   num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attenton_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id, eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = 'attn_dec_inf')
    
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions

# Decoding RNN
def decoder_rnn(decoder_embedded_input, 
                decoder_embeddings_matrix, 
                encoder_state, 
                num_words, 
                sequence_length, 
                rnn_size, 
                num_layers, 
                word2int, 
                keep_prob, 
                batch_size):
    with tf.variable_scope('decoding') as decoding_scope: 
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        
        training_predictions = decode_training_set(encoder_state, 
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'], 
                                           sequence_length - 1, 
                                           num_words,
                                           decoding_scope, 
                                           output_function,
                                           keep_prob,
                                           batch_size)
        
    return training_predictions, test_predictions


# Building seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                     answers_num_words + 1,
                                                     encoder_embedding_size,
                                                     initializer= tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input,
                                rnn_size,
                                num_layers,
                                keep_prob,
                                sequence_length)
    
    preprocessed_targets = preprocess_targets(targets, 
                                              questionswords2int, 
                                              batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    training_pred, test_pred = decoder_rnn(decoder_embedded_input, 
                                           decoder_embeddings_matrix, 
                                           encoder_state, 
                                           questions_num_words, 
                                           sequence_length, 
                                           rnn_size, 
                                           num_layers, 
                                           questionswords2int, 
                                           keep_prob, 
                                           batch_size)
    
    return training_pred, test_pred


############ PART 3 - TRAINING THE SEQ2SEQ MODEL ##############
# Setting the hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# Getting the shape of the input tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_prediction, test_prediction = seq2seq_model(tf.reverse(inputs, [-1]),
                                                     targets,
                                                     keep_prob,
                                                     batch_size,
                                                     sequence_length,
                                                     len(answords2int),
                                                     len(queswords2int),
                                                     encoding_embedding_size,
                                                     decoding_embedding_size,
                                                     rnn_size,
                                                     num_layers,
                                                     queswords2int)

