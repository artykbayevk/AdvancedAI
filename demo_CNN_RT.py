# In[1]

import os
import pandas as pd
from keras.models import  load_model
from keras.utils import to_categorical
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[2]
NUM_WORDS=30000
DATA = pd.read_csv(os.path.join('data/RT', 'dataset.tsv'), sep='\t')
x_raw = DATA.Phrase
y_raw = DATA.Sentiment
x_original_train, x_original_test, y_original_train, y_original_test = train_test_split(x_raw, y_raw,
                                                                                        test_size = 0.2, random_state = 42)
word2vec = KeyedVectors.load_word2vec_format(os.path.join('weights','GoogleNews-vectors-negative300.bin.gz' )
                                             ,binary=True)
model = load_model('weights/CNN-RT.hdf5')
# In[3]

tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
tokenizer.fit_on_texts(x_original_train)
train = pad_sequences(tokenizer.texts_to_sequences(x_original_train))
valid = pad_sequences(tokenizer.texts_to_sequences(x_original_test), maxlen=train.shape[1])
y_val = to_categorical(y_original_test)

# In[4]
pred_X = model.predict(valid).argmax(axis=-1)
print("Accuracy: {}".format(accuracy_score(y_original_test, pred_X)))


# In[10]

sequences_test=tokenizer.texts_to_sequences(['predicting sentiment for text','it is great',
                                             'is it awful','it is good','too nice and good'])
pred_classes = model.predict(pad_sequences(sequences_test,maxlen=48)).argmax(axis=-1)
print(pred_classes)
