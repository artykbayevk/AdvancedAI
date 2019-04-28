#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
from keras.models import load_model
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[2]:
NUM_WORDS=30000
DATA = pd.read_csv(os.path.join('data','Twitter', 'training.1600000.processed.noemoticon.csv'),
                   encoding = "ISO-8859-1", names = ["target", "ids", "data", 'flag', "user", "text"])
word2vec = KeyedVectors.load_word2vec_format(os.path.join('weights','GoogleNews-vectors-negative300.bin.gz' ),
                                             binary=True)
model = load_model('weights/CNN-Twitter.h5')
model.summary()


# In[3]:
mapping = {0:0, 4:1}

x_raw = DATA.text
y_raw = DATA.target.apply(lambda x:mapping[x])
x_original_train, x_original_test, y_original_train, y_original_test = train_test_split(x_raw, y_raw,
                                                                                        test_size = 0.2, random_state = 42)

tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
tokenizer.fit_on_texts(x_original_train)
# In[4]

X_test = pad_sequences(tokenizer.texts_to_sequences(x_original_test),117)
pred_X = model.predict(X_test).argmax(axis=-1)
print('Accuracy: {}'.format(accuracy_score(pred_X, y_original_test)))

# In[5]
texts = ['predicting sentiment for text','it is great','is it awful','it is good','too nice and good']
sequences_test=tokenizer.texts_to_sequences(texts)
test_data = pad_sequences(sequences_test,maxlen=117)
pred = model.predict(test_data)
print(pred)
pred_classes = pred.argmax(axis=-1)
print(pred_classes)