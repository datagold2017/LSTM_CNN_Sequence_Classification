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

# # fix random seed for reproducibility
# numpy.random.seed(7)
#
# # load the data set but only keep the top n words, zero the rest
top_words = 500
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# returns a compiled model identical to the previous one
model = load_model('../saved_models/0.0.1_model.h5')

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of Disk-Loaded Model: %.2f%%" % (scores[1]*100))

# Mar 7, 2017 Accuracy of Disk-Loaded Model: 51.17%