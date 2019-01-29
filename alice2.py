import numpy
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, encoding="utf8").read()
raw_text = re.sub('[^a-zA-Z]', ' ', raw_text.lower()).split()

words = sorted(list(set(raw_text)))
word_to_int = dict((c, i) for i, c in enumerate(words))

n_char = len(raw_text)
n_vocab = len(words)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_char - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([word_to_int[word] for word in seq_in])
	dataY.append(word_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#model.fit(X, y, epochs=10, batch_size=128, callbacks=callbacks_list)

# load the network weights
filename = "weights-improvement-01-6.3246.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_words = dict((i, c) for i, c in enumerate(words))

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ' '.join([int_to_words[value] for value in pattern]), "\"")
# generate characters
res = ""
for i in range(10):
	print([int_to_words[value] for value in pattern])
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=1)	
	index = numpy.argmax(prediction)
	#print(prediction[0][index])
	result = int_to_words[index]
	seq_in = [int_to_words[value] for value in pattern]
	#sys.stdout.write(result)
	res += result;
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print(res)
print("\nDone.")

