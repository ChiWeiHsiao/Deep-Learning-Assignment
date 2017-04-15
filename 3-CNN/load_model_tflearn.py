import _pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import local_response_normalization

def unpickle(filename):
	file_object = open(filename, 'rb')
	data = _pickle.load(file_object, encoding='latin1') 
	file_object.close()
	return data
'''
def to_categorical(y, nb_classes):
	y = np.asarray(y, dtype='int32')
	if not nb_classes:
		nb_classes = np.max(y)+1
		Y = np.zeros((len(y), nb_classes))
		Y[np.arange(len(y)),y] = 1.
		return Y
'''
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
  #print('Y:', Y.shape)
  return label_name, X, Y, X_test, Y_test 
  

if __name__ == '__main__':
  # Load data
  label_name, X, Y, X_test, Y_test = get_cifar_10()

  # Real-time data augmentation
  img_aug = ImageAugmentation()
  img_aug.add_random_flip_leftright()
  img_aug.add_random_rotation(max_angle=25.)

  params = {
    'conv_filter': 5,
    'pool_width': 3,
    'pool_stride': 2,
    'epoch': 50,
    'id': 'cnn_4'
  }
  # Build CNN
  input_data = input_data(shape=[None, 32, 32, 3], data_augmentation=img_aug)
  conv1 = conv_2d(input_data, 64, params['conv_filter'], activation='relu', regularizer='L2')
  pool1 = max_pool_2d(conv1, params['pool_width'], params['pool_stride'])
  lrn1 = local_response_normalization(pool1)

  conv2 = conv_2d(lrn1, 64, params['conv_filter'], activation='relu', regularizer='L2')
  pool2 = max_pool_2d(conv2, params['pool_width'], params['pool_stride'])
  lrn2 = local_response_normalization(pool2)

  conv3 = conv_2d(lrn2, 128, params['conv_filter'], activation='relu', regularizer='L2')
  pool3 = max_pool_2d(conv3, params['pool_width'], params['pool_stride'])
  lrn3 = local_response_normalization(pool3)

  flat = flatten(lrn3) 

  fully1 = fully_connected(lrn3, 384, activation='relu')
  drop1 = dropout(fully1, 0.5)
  fully2 = fully_connected(drop1, 384/2, activation='relu')
  drop2 = dropout(fully2, 0.5)
  fully3 = fully_connected(drop2, 10, activation='softmax')
  network = regression(fully3, optimizer='adam',
											 loss='categorical_crossentropy',
											 learning_rate=0.001, name='Target')

	# Define model
  model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='../log/')

	# Predict
  model.load('../log/model_4/'+params['id']+'.tflearn')
  test_x = []
  test_x.append( np.reshape(X_test[0], [1, 32, 32, 3]) )
  test_x.append( np.reshape(X_test[1], [1, 32, 32, 3]))
  print('Predict X[0]: ', np.argmax(model.predict(test_x[0])))
  print('Truth Y[0]: ', np.argmax(Y_test[0]))
  print('Predict X[0]: ', np.argmax(model.predict(test_x[1])))
  print('Truth Y[0]: ', np.argmax(Y_test[1]))

	# image
	img_pool1 = tflearn.helpers.summarizer.summarize (pool1, 'image', name, summary_collection='img_pool1')
	img_pool2 = tflearn.helpers.summarizer.summarize (pool2, 'image', name, summary_collection='img_pool2')
	img_pool3 = tflearn.helpers.summarizer.summarize (pool3, 'image', name, summary_collection='img_pool3')


  # Get weights
  #print("conv1 layer weights[0]:")
  #print(model.get_weights(conv1.W)[0])


