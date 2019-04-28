# In[1]
import os
import pandas as pd
from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sklearn.metrics import accuracy_score
train = pd.read_csv("data/RT/dataset.tsv", sep="\t")
test = pd.read_csv("data/RT/test.tsv", sep="\t")
LSTM_RT = load_model('weights/LSTM-RT.h5')


# In[3]
phrases = train.append(test)['Phrase'].tolist()
kerasTok = text.Tokenizer()
kerasTok.fit_on_texts(phrases)
seq = kerasTok.texts_to_sequences(phrases)
X = sequence.pad_sequences(seq, 60)
train_data,test_data =train_test_split(train, test_size=0.3,random_state=42)
test_data_text = test_data['Phrase']
test_data_sentiment =  test_data['Sentiment']
test_X = sequence.pad_sequences(kerasTok.texts_to_sequences(test_data_text), 60)
test_Y = test_data_sentiment

# In[2]
pred_classes = LSTM_RT.predict(test_X).argmax(axis=-1)
print('Accuracy: {}'.format(accuracy_score(pred_classes, test_Y)))


# In[1]
texts = ['predicting sentiment for text','it is great','is it awful','it is good','too nice and good']
texts_ = kerasTok.texts_to_sequences(texts)
X_test = sequence.pad_sequences(texts_, 60)
pred = LSTM_RT.predict(X_test)
pred_classes = pred.argmax(axis=-1)
print(pred_classes)

