import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re

data = pd.read_csv('Sentiment.csv')

#keep the necessary columns only
data = data[['text','sentiment']]

#function to remove unwanted characters
def pre_process_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

data['text'] = data['text'].apply(pre_process_data)

#use Tensorflow’s tokenizer to tokenize our dataset, and Tensorflow’s pad_sequences to pad our sequences.

max_features = 2000
tokenizer = Tokenizer(num_words = max_features, split = ' ')
tokenizer.fit_on_texts(data['text'].values)
x = tokenizer.texts_to_sequences(data['text'].values)
x = pad_sequences(x,28)

y = pd.get_dummies(data['sentiment']).values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length =x.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout = 0.3, recurrent_dropout = 0.2, return_sequences = True))
model.add(LSTM(128,recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

batch_size = 512

model.fit(x_train, y_train, epochs = 10, batch_size=batch_size, validation_data=(x_test, y_test))