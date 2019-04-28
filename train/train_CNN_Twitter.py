#!/usr/bin/env python
# coding: utf-8

# In[1]
import os
import numpy as np
import pandas as pd
import pickle as pkl
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from keras.utils import to_categorical
from keras.layers.core import Reshape, Flatten
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate

# In[2]

STOPWORDS = stopwords.words("english")
STEMMER = SnowballStemmer("english")

DATA_FILE_PATH = os.path.join('data/Twitter/training.1600000.processed.noemoticon.csv')
DATA = pd.read_csv(DATA_FILE_PATH, encoding = "ISO-8859-1", names =
    ["target", "ids", "data", 'flag', "user", "text"])

WORD2VEC_PATH = os.path.join('weights/GoogleNews-vectors-negative300.bin.gz')
word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
mapping = {0:0, 4:1}

x_raw = DATA.text
y_raw = DATA.target.apply(lambda x:mapping[x])


# In[3]
x_original_train, x_original_test, y_original_train, y_original_test = train_test_split(x_raw,
                                                                                        y_raw, test_size = 0.2, random_state = 42)

NUM_WORDS=30000
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
tokenizer.fit_on_texts(x_original_train)

sequences_train = tokenizer.texts_to_sequences(x_original_train)
sequences_valid=tokenizer.texts_to_sequences(x_original_test)

words = tokenizer.word_index

X_train = pad_sequences(sequences_train)
X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])
y_train = to_categorical(y_original_train)
y_val = to_categorical(y_original_test)

EMBEDDING_DIM=300
vocabulary_size=min(len(words)+1,NUM_WORDS)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

# In[4]

for word, i in words.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word2vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)


embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)

sequence_length = X_train.shape[1]
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5

# In[5]

def cnn(sequence_length):
    inputs = Input(shape=(sequence_length,))
    embedding = embedding_layer(inputs)
    reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

    conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
    maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
    maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten()(merged_tensor)
    # reshape = Reshape((3*num_filters,))(flatten)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)

    # this creates a model that includes
    model = Model(inputs, output)
    return model

model = cnn(sequence_length)
model.summary()


# In[6]

adam = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc', 'mae'])
history = model.fit(X_train, y_train, batch_size=1000, epochs=100, verbose=1, validation_data=(X_val, y_val))  # starts training


# In[7]
model.save('train_data/CNN_TWITTER.h5')
with open('train_data/CNN_TWITTER_HISTORY.pkl', 'wb') as file_pi:
    pkl.dump(history.history, file_pi)


