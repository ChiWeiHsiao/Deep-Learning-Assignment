''' data utility functions '''
import _pickle
import numpy as np

def unpickle(filename):
	file_object = open(filename, 'rb')
	data = _pickle.load(file_object, encoding='latin1') 
	file_object.close()
	return data


def shuffle(*pairs):
  pairs = list(pairs)
  for i, pair in enumerate(pairs):
    pairs[i] = np.array(pair)
  p = np.random.permutation(len(pairs[0]))
  return tuple(pair[p] for pair in pairs)

def to_categorical(y, nb_classes):
  y = np.asarray(y, dtype='int32')
  Y = np.zeros((len(y), nb_classes))
  Y[np.arange(len(y)),y] = 1.
  return Y

def get_cifar_10(): 
	label_name = unpickle('../data/CIFAR-10/batches.meta')['label_names']
	X, Y = [], []
	for i in range(1,6):
		batch = unpickle('../data/CIFAR-10/data_batch_'+str(i))
		if i == 1:
			X= batch['data']
			Y = batch['labels']
		else:
			X = np.append(X, batch['data'], axis=0)
			Y = np.append(Y, batch['labels'], axis=0)
	# seems no need for shuffle
	X, Y = shuffle(X, Y)
	test_batch = unpickle('../data/CIFAR-10/test_batch')
	X_test, Y_test = test_batch['data'], test_batch['labels'] 
	# Reshape X: (50000, 3072) -> (50000, 32, 32, 3)
	X = np.dstack((X[:, :1024], X[:, 1024:2048], X[:, 2048:])) # (50000, 1024, 3)
	X = np.reshape(X, [-1, 32, 32, 3])  # (50000, 32, 32, 3)
	X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))
	X_test = np.reshape(X_test, [-1, 32, 32, 3])
	# one-hot
	Y = to_categorical(Y, 10)
	Y_test = to_categorical(Y_test, 10)
	print('X:', X.shape)
	print('Y:', Y.shape)
	return label_name, X, Y, X_test, Y_test 

class GetBatch():
	def __init__(self, X, Y, batch_size):
		self.X_batch, self.Y_batch = [], []
		self.state = 0
		batch_size = batch_size
		self.total_batch = int(X.shape[0] / batch_size)
		start = 0

		for i in range(self.total_batch):
			self.X_batch.append(X[start:start+batch_size, :, :])
			self.Y_batch.append(Y[start:start+batch_size, :])
			start += batch_size

	def next_batch(self):
		self.state += 1
		self.state %= self.total_batch
		next_batch = self.X_batch[self.state], self.Y_batch[self.state]
		return next_batch

