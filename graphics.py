# In[1]

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# In[2] LSTM trained on Rotten Tomatoes
with open('history/LSTM_RT_HISTORY.pkl', 'rb') as f:
    lstm_rt_history = pkl.load(f)

lstm_rt_loss = lstm_rt_history['loss']
lstm_rt_acc = lstm_rt_history['acc']
lstm_rt_mean_absolute_error = lstm_rt_history['mean_absolute_error']


plt.style.use("ggplot")
plt.plot(np.arange(0, 100), lstm_rt_loss,label = 'categorical crossentropy')
plt.plot(np.arange(0, 100), lstm_rt_mean_absolute_error,label = 'mean absolute error')
plt.title("Categorical Crossentropy and MAE. LSTM-RT")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()


plt.style.use("ggplot")
plt.plot(np.arange(0, 100), lstm_rt_acc,label = 'categorical accuracy')
plt.title("Categorical Accuracy. LSTM-RT")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()


# In[2] LSTM trained on Rotten Tomatoes
with open('history/CNN_RT_HISTORY.pkl', 'rb') as f:
    cnn_rt_history = pkl.load(f)

cnn_rt_loss = cnn_rt_history['loss']
cnn_rt_acc = cnn_rt_history['acc']
cnn_rt_mean_absolute_error = cnn_rt_history['mean_absolute_error']

cnn_rt_val_loss = cnn_rt_history['val_loss']
cnn_rt_val_acc = cnn_rt_history['val_acc']
cnn_rt_mae = cnn_rt_history['val_mean_absolute_error']

plt.style.use("ggplot")
plt.plot(np.arange(0, 1000), cnn_rt_loss,label = 'categorical crossentropy')
plt.plot(np.arange(0, 1000), cnn_rt_mean_absolute_error,label = 'mean absolute error')
plt.plot(np.arange(0, 1000), cnn_rt_mae,label = 'val mean absolute error')
plt.title("Categorical Crossentropy and MAE. CNN-RT")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()


plt.style.use("ggplot")
plt.plot(np.arange(0, 1000), cnn_rt_acc,label = 'categorical accuracy')
plt.plot(np.arange(0, 1000), cnn_rt_val_acc,label = 'val categorical accuracy')
plt.title("Categorical Accuracy. CNN-RT")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

# In[5] LSTM trained on Twitter 140

with open('history/LSTM_RT_HISTORY.pkl', 'rb') as f:
    lstm_tw_history = pkl.load(f)
print(lstm_tw_history.keys())

lstm_tw_loss = lstm_tw_history['loss']
lstm_tw_acc = lstm_tw_history['acc']
lstm_tw_mean_absolute_error = lstm_tw_history['mean_absolute_error']


plt.style.use("ggplot")
plt.plot(np.arange(0, 100), lstm_tw_loss,label = 'categorical crossentropy')
plt.plot(np.arange(0, 100), lstm_tw_mean_absolute_error,label = 'mean absolute error')
plt.title("Categorical Crossentropy and MAE. LSTM-Twitter140")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

plt.style.use("ggplot")
plt.plot(np.arange(0, 100), lstm_tw_acc,label = 'categorical accuracy')
plt.title("Categorical Accuracy. LSTM-Twitter140")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()


# In[6]

with open('history/CNN_RT_HISTORY.pkl', 'rb') as f:
    cnn_rt_history = pkl.load(f)

cnn_rt_loss = cnn_rt_history['loss']
cnn_rt_acc = cnn_rt_history['acc']
cnn_rt_mean_absolute_error = cnn_rt_history['mean_absolute_error']

cnn_rt_val_loss = cnn_rt_history['val_loss']
cnn_rt_val_acc = cnn_rt_history['val_acc']
cnn_rt_mae = cnn_rt_history['val_mean_absolute_error']

plt.style.use("ggplot")
plt.plot(np.arange(0, 100), cnn_rt_loss,label = 'categorical crossentropy')
plt.plot(np.arange(0, 100), cnn_rt_mean_absolute_error,label = 'mean absolute error')
plt.plot(np.arange(0, 100), cnn_rt_mae,label = 'val mean absolute error')
plt.title("Categorical Crossentropy and MAE. CNN-Twitter140")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()


plt.style.use("ggplot")
plt.plot(np.arange(0, 100), cnn_rt_acc,label = 'categorical accuracy')
plt.plot(np.arange(0, 100), cnn_rt_val_acc,label = 'val categorical accuracy')
plt.title("Categorical Accuracy. CNN-Twitter140")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()