# In[1]


import os
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[3]

# loading LSTM model trained on rotten tomatoes dataset
LSTM_RT_model = load_model('weights/RT/LSTM/LSTM-RT.h5')
LSTM_Twitter_model = load_model('weights/Twitter/LSTM/LSTM-Twitter.hdf5')
CNN_RT_model = load_model('weights/RT/CNN/CNN-RT.hdf5')
CNN_Twitter_model = load_model('weights/Twitter/CNN/CNN-Twitter.h5')

# In[3]


def LSTM_RT_tokenizer():
    train = pd.read_csv("data/RT/train.tsv", sep="\t")
    test = pd.read_csv("data/RT/test.tsv", sep="\t")

    phrases = train.append(test)["Phrase"].tolist()
    kerasTok = text.Tokenizer()
    kerasTok.fit_on_texts(phrases)

    return kerasTok

def LSTM_TWITTER_tokenizer():
    train = pd.read_csv("data/Twitter/train.csv")
    test = pd.read_csv("data/Twitter/test.csv")
    phrases = train.append(test)["text"].tolist()
    kerasTok = text.Tokenizer()
    kerasTok.fit_on_texts(phrases)

    return kerasTok

def CNN_RT_tokenizer():
    DATA = pd.read_csv(os.path.join('data/RT', 'train.tsv'), sep='\t')
    x_raw = DATA.Phrase
    y_raw = DATA.Sentiment

    x_original_train, x_original_test, y_original_train, y_original_test = train_test_split(x_raw, y_raw, test_size=0.2,
                                                                                            random_state=42)
    NUM_WORDS = 30000
    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)
    tokenizer.fit_on_texts(x_original_train)

    return tokenizer

def CNN_TWITTER_tokenizer():
    DATA_FILE_PATH = os.path.join('data', 'Twitter', 'training.1600000.processed.noemoticon.csv')
    DATA = pd.read_csv(DATA_FILE_PATH, encoding="ISO-8859-1", names=["target", "ids", "data", 'flag', "user", "text"])

    x_raw = DATA.text
    y_raw = DATA.target
    new_target_encoding = {0: 0, 4: 1}
    y_raw = y_raw.apply(lambda x: new_target_encoding[x])

    x_original_train, x_original_test, y_original_train, y_original_test = train_test_split(x_raw, y_raw, test_size=0.2,
                                                                                            random_state=42)

    NUM_WORDS = 30000
    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)
    tokenizer.fit_on_texts(x_original_train)

    return tokenizer


# In[4]


def LSTM_RT(text):
    tags = {0: 'negative',
               1: 'somewhat negative',
               2: 'neutral',
               3: 'somewhat positive',
               4: 'positive'}

    tokenizer = LSTM_RT_tokenizer()
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(text), 60)

    pred_classes = LSTM_RT_model.predict(X_test).argmax(axis=-1)

    results = {}

    for txt, label in zip(text, pred_classes):
        results[txt] = tags[label]

    return results

def LSTM_TWITTER(text):
    tags = {0: 'negative', 1: 'positive'}
    tokenizer = LSTM_TWITTER_tokenizer()
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(text), 60)
    pred_classes = LSTM_Twitter_model.predict(X_test).argmax(axis=-1)
    results = {}

    for txt, label in zip(text, pred_classes):
        results[txt] = tags[label]

    return results

def CNN_RT(text):
    tags = {0: 'negative',
               1: 'somewhat negative',
               2: 'neutral',
               3: 'somewhat positive',
               4: 'positive'}
    tokenizer = CNN_RT_tokenizer()
    test_data = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=48)
    pred_classes = CNN_RT_model.predict(test_data).argmax(axis=-1)
    results = {}

    for txt, label in zip(text, pred_classes):
        results[txt] = tags[label]

    return results

def CNN_TWITTER(text):
    tags = {0: 'negative', 1: 'positive'}
    tokenizer = CNN_TWITTER_tokenizer()
    test_data = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=117)
    pred_classes = CNN_Twitter_model.predict(test_data).argmax(axis=-1)
    results = {}

    for txt, label in zip(text, pred_classes):
        results[txt] = tags[label]

    return results


# In[5]


texts = ['predicting sentiment for text', 'it is great', 'is it awful', 'it is good', 'too nice and good']
res_1 = LSTM_RT(texts)
res_2 = LSTM_TWITTER(texts)
res_3 = CNN_RT(texts)
res_4 = CNN_TWITTER(texts)


print(res_1)
print(res_2)
print(res_3)
print(res_4)


# In[6]
print('Given texts:________________________________ ')
for text in texts:
    print(text)
print('____________________________________________')
print('LSTM trained on DATASET1:', end="[")
for text in texts:
    print(res_1[text], end=", ")
print(']')
print('LSTM trained on DATASET2:', end="[")
for text in texts:
    print(res_2[text], end=", ")
print(']\n')

print('CNN  trained on DATASET1:', end="[")
for text in texts:
    print(res_3[text], end=", " )
print(']')
print('CNN  trained on DATASET2:', end="[")
for text in texts:
    print(res_3[text], end=", ")
print(']')
