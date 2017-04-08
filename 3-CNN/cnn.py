import _pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


def unpickle(filename):
	file_object = open(filename, 'rb')
	data = _pickle.load(file_object, encoding='latin1') 
	file_object.close()
	return data


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
      #Y = Y + batch['labels']
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
  Y_test = to_categorical(Y, 10)
  print('X:', X.shape)
  print('Y:', Y.shape)
  return label_name, X, Y, X_test, Y_test 
  

def cnn():
  print('cnn')


if __name__ == '__main__':
  # Load data
  label_name, X, Y, X_test, Y_test = get_cifar_10()
  # Build CNN
  network = input_data(shape=[None, 32, 32, 3], name='InputData')
  network = conv_2d(network, 32, 3, activation='relu')
  network = max_pool_2d(network, 2)
  network = conv_2d(network, 64, 3, activation='relu')
  network = conv_2d(network, 64, 3, activation='relu')
  network = max_pool_2d(network, 2)
  network = fully_connected(network, 512, activation='relu')
  network = dropout(network, 0.5)
  network = fully_connected(network, 10, activation='softmax')
  network = regression(network, optimizer='adam',
											 loss='categorical_crossentropy',
											 learning_rate=0.001)

	# Train using classifier
  model = tflearn.DNN(network, tensorboard_verbose=0)
  model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
						show_metric=True, batch_size=96, run_id='cifar10_cnn')
