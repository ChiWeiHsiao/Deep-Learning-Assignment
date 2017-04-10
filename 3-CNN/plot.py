import tflearn

  input_data = input_data(shape=[None, 32, 32, 3], data_augmentation=img_aug)
  conv1 = conv_2d(input_data, nb_filter=64, filter_size=3, activation='relu', regularizer='L2')
  pool1 = max_pool_2d(conv1, kernel_size=3, strides=2)
  lrn1 = local_response_normalization(pool1)

  conv2 = conv_2d(lrn1, 64, 3, activation='relu', regularizer='L2')
  pool2 = max_pool_2d(conv2, 3, strides=2)
  lrn2 = local_response_normalization(pool2)

  conv3 = conv_2d(lrn2, 128, 3, activation='relu', regularizer='L2')
  pool3 = max_pool_2d(conv3, 3, strides=2)
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

  model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='../log/') 
  model.load('model_2.tflearn')
  model.predict()
