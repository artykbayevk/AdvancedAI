# In[1]

import os
import pandas as pd
from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence, text
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[1]
LSTM_Twitter = load_model('weights/LSTM-Twitter.hdf5')
LSTM_Twitter.summary()

# In[2]
train = pd.read_csv("data/Twitter/train.csv")
test = pd.read_csv("data/Twitter/test.csv")

phrases= train.append(test)["text"].tolist()
kerasTok = text.Tokenizer()
kerasTok.fit_on_texts(phrases)
X = sequence.pad_sequences(kerasTok.texts_to_sequences(phrases), 60)
mapping = {0:0, 4:1}
test_X = sequence.pad_sequences(kerasTok.texts_to_sequences(train['text'][:100000]), 60) # we take only 100K samples, because there are 1 million samples for testing, it takes a lot of times
y_raw = train['target'].apply(lambda x:mapping[x])[:100000]

# In[3]

pred_X = LSTM_Twitter.predict(test_X).argmax(axis=-1)
print('Accuracy: {}'.format(accuracy_score(pred_X, y_raw)))


# In[4]
texts = ['predicting sentiment for text','it is great','is it awful','it is good','too nice and good']
texts_ = kerasTok.texts_to_sequences(texts)
X_test = sequence.pad_sequences(texts_, 60)
pred = LSTM_Twitter.predict(X_test)
print(pred)
pred_classes = pred.argmax(axis=-1)
print(pred_classes)
