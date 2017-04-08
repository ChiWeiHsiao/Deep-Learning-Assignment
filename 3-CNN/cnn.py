import _pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


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
  

if __name__ == '__main__':
  # Load data
  label_name, X, Y, X_test, Y_test = get_cifar_10()
  # Build CNN
    # ref: http://terrence.logdown.com/posts/1296387
  input_data = input_data(shape=[None, 32, 32, 3])
  conv1 = conv_2d(input_data, nb_filter=64, filter_size=3, activation='relu', regularizer='L2')
  pool1 = max_pool_2d(conv1, kernel_size=2, strides=[1,2,2,1])
  lrn1 = local_response_normalization(pool1)

  conv2 = conv_2d(lrn1, 128, 3, activation='relu', regularizer='L2')
  pool2 = max_pool_2d(conv2, 2, strides=[1,2,2,1])
  lrn2 = local_response_normalization(pool2)

  fully1 = fully_connected(lrn2, 512, activation='relu')
  #fully1 = dropout(fully1, 0.8)
  fully2 = fully_connected(fully1, 1024, activation='relu')
  fully2 = dropout(fully2, 0.8)
  fully3 = fully_connected(fully2, 10, activation='softmax')
  network = regression(fully3, optimizer='adam',
											 loss='categorical_crossentropy',
											 learning_rate=0.001, name='Target')

	# Train using classifier
  model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='../log/',)
  model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
						snapshot_step=100, show_metric=True, batch_size=128, run_id='cnn_1')

  model.save('model_1.tflearn')

  # Get weights
  print("conv1 layer weights[0]:")
  print(model.get_weights(conv1.W)[0])
