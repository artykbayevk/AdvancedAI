#!/usr/bin/env python
# coding: utf-8

# In[25]:

import numpy as np
import pandas as pd
import pickle as pkl
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import Embedding
from keras.utils import to_categorical
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Reshape, Flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout,concatenate



# In[2]:

DATA = pd.read_csv('data/RT/dataset.tsv', sep='\t')
word2vec = KeyedVectors.load_word2vec_format('weights/GoogleNews-vectors-negative300.bin.gz', binary=True)

print(DATA.head())
print(DATA.columns)
print(DATA.Sentiment.value_counts())

x_raw = DATA.Phrase
y_raw = DATA.Sentiment

x_original_train, x_original_test, y_original_train, y_original_test = train_test_split(x_raw,
                                                                                        y_raw, test_size = 0.2, random_state = 42)


NUM_WORDS=30000
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
tokenizer.fit_on_texts(x_original_train)
sequences_train = tokenizer.texts_to_sequences(x_original_train)
sequences_valid=tokenizer.texts_to_sequences(x_original_test)
word_index = tokenizer.word_index

X_train = pad_sequences(sequences_train)
X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])
y_train = to_categorical(y_original_train)
y_val = to_categorical(y_original_test)


# In[12]:


EMBEDDING_DIM=300
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

for word, i in word_index.items():
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


# In[15]:


sequence_length = X_train.shape[1]
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5



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
    dropout = Dropout(drop)(flatten)
    output = Dense(units=5, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = Model(inputs, output)
    return model

model = cnn(sequence_length)


model.summary()


# In[18]:

adam = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc', 'mae','categorical_accuracy'])
history = model.fit(X_train, y_train, batch_size=1000, epochs=1000, verbose=1, validation_data=(X_val, y_val))  # starts training


# In[3]
model.save('train/train_data/CNN_RT.hdf5')
with open('train/train_data/CNN_RT_TWITTER.pkl', 'wb') as file_pi:
    pkl.dump(history.history, file_pi)


