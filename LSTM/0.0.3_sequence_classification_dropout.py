# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model

# fix random seed for reproducibility
numpy.random.seed(7)

# load the data set but only keep the top n words, zero the rest
# top_words = 500
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Partial data
train_size = numpy.size(X_train[:, 0])
test_size = numpy.size(X_test[:, 0])
# print(train_size, test_size)
partial = .1
train_partial = int(train_size * partial)
test_partial = int(test_size * partial)

# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length, dropout=0.2))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# model.fit(X_train[0:train_partial, :], y_train[0:train_partial], nb_epoch=1, batch_size=64)

# First evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
# scores = model.evaluate(X_test[0:test_partial, :], y_test[0:test_partial], verbose=0)
print("Accuracy of Run Model: %.2f%%" % (scores[1]*100))

model.save('../saved_models/0.0.3_model.h5')

del model  # deletes the existing model

# returns a compiled model identical to the previous one
model = load_model('../saved_models/0.0.3_model.h5')

# Final evaluation of the model
disk_loaded_scores = model.evaluate(X_test, y_test, verbose=0)
# scores = model.evaluate(X_test[0:test_partial, :], y_test[0:test_partial], verbose=0)
print("Accuracy of Disk-Loaded Model: %.2f%%" % (disk_loaded_scores[1]*100))

# Mar 7, 2017 Accuracy of Run Model: 84.31%