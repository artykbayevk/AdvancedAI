# In[1]

import numpy as np
import pandas as pd
import pickle as pkl
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, Input, Dropout, LSTM, Bidirectional

# In[2]
train = pd.read_csv('data/Twitter/train.csv')
test = pd.read_csv('data/Twitter/test.csv')
all_data = train.append(test)
all_data.head()

texts = all_data["text"].tolist()

kerasTok = text.Tokenizer(lower=True,split=' ',filters='[0-9]!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
kerasTok.fit_on_texts(texts)
all_phrases = kerasTok.texts_to_sequences(texts)

X = sequence.pad_sequences(all_phrases, 60)
X_train = X[:train.shape[0], :]
X_test = X[train.shape[0]:, :]

# In[2]

vocab_size = len(kerasTok.word_counts)
embed_size = 200
maxLen = 60

Y_train = np.array(train.target)
encode = OneHotEncoder(sparse=False)
Y_train_1hot = encode.fit_transform(np.reshape(Y_train, (Y_train.shape[0], 1)))

# In[3]

def lstm(input_shape, vocab_len, embed_size):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = Embedding(vocab_len + 1, embed_size)

    embeddings = embedding_layer(sentence_indices)

    X = Bidirectional(LSTM(units=128, return_sequences=True))(embeddings)
    X = Dropout(rate=0.6)(X)

    X = Bidirectional(LSTM(units=64))(X)
    X = Dropout(rate=0.3)(X)

    X = Dense(units=2, activation='softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)
    return model

model = lstm((maxLen,), vocab_size, embed_size)
model.summary()

# In[4]

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, Y_train_1hot, batch_size=128, epochs=5)

model.save('train/train_data/LSTM_Twitter.h5')
with open('train/train_data/LSTM_Twitter_HISTORY.pkl', 'wb') as file_pi:
    pkl.dump(history.history, file_pi)
