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
	first_image = X[0]
	first_label = to_categorical( np.reshape(Y[0], [1,1]), 10)
	print('first image: ', first_image.shape)
	first_image = np.dstack((first_image[:1024], first_image[1024:2048], first_image[2048:]))
	print('first image: ', first_image.shape)
	first_image = np.reshape(first_image, [1, 32, 32, 3])
	print('first image: ', first_image.shape)
	print('first_label: ', first_label)
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
	return label_name, X, Y, X_test, Y_test, first_image, first_label
