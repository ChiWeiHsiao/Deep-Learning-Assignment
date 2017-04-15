from __future__ import print_function
import numpy as np
import tensorflow as tf
import json
from util import get_cifar_10

params = {
  'conv_filter': 3,
  'conv_stride': 1,
  'pool_kernel': 3,
  'pool_stride': 2,
  'dropout': 0.1, # i.e. keep_prob = 1-params['dropout']
  'epoch': 5,
  'batch_size': 128,
}

eid = 'try'
log = {
  'experiment_id': eid,
  'train_accuracy_per_epoch': [],
  'test_accuracy_per_epoch': [],
  'original_image': [],
  'feature_map_0': { 'conv1': [], 'pool1': [], 'lrn1': [], 'conv2': [], 'pool2': [], 'lrn2': []},
  'feature_map_1': { 'conv1': [], 'pool1': [], 'lrn1': [], 'conv2': [], 'pool2': [], 'lrn2': []},
}
logfile = 'statistics/'+eid+'.json'

print('id: ', eid)
print('num of epochs:', params['epoch'])

''' Load data '''
label_name, train_X, train_Y, test_X, test_Y = get_cifar_10()
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = train_X.shape[1:4]
NUM_CLASS = train_Y.shape[1]
TRAIN_SIZE = train_X.shape[0]

log['original_image'] = train_X[0].tolist()
first_image = np.reshape(train_X[0], (1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
first_label = np.reshape(train_Y[0], (1, NUM_CLASS)) # add one dim, to make first_label as a batch


''' tf Wrapper '''
def add_conv(in_tensor, out_size):
  biases_initializer = tf.constant_initializer(0.)
  weights_initializer = tf.uniform_unit_scaling_initializer(seed=None, dtype=tf.float32)

  conv = tf.contrib.layers.conv2d(in_tensor, out_size, kernel_size=params['conv_filter'], stride=params['conv_stride'], activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, padding='SAME')
  return tf.nn.relu(conv)

def add_maxpool(in_tensor):
  pool = tf.contrib.layers.max_pool2d(in_tensor, kernel_size=params['pool_kernel'], stride=params['pool_stride'], padding='SAME')
  return pool

def add_lrn(in_tensor):
  return tf.nn.lrn(in_tensor, depth_radius=5, bias=1.0, alpha=0.001/9.0, beta=0.75)

def add_fully(in_tensor, n_out):
  biases_initializer = tf.constant_initializer(0.)
  weights_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=tf.float32)
  fc = tf.contrib.layers.fully_connected(in_tensor, n_out, activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, trainable=True)
  return tf.nn.relu(fc)


'''Define model'''
# Graph input
x = tf.placeholder('float', [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
y_truth = tf.placeholder('float', [None, NUM_CLASS])
# Conv 1: [-1, 32, 32, 1] -> [-1, 32, 32, 64]
input_layer = tf.reshape(x, shape=[-1, 32, 32, 3])
conv1 = add_conv(input_layer, 64)
# Pool 1: [-1, 32, 32, 64] -> [-1, 16, 16, 64] (if stride=2)
pool1 = add_maxpool(conv1)
lrn1 = add_lrn(pool1)
# Conv 2: [-1, 16, 16, 64] -> [-1, 16, 16, 64]
conv2 = add_conv(lrn1, 64) #64 or 128
# Pool 2: [-1, 16, 16, 64] -> [-1, 8, 8, 64]
pool2 = add_maxpool(conv2)
lrn2 = add_lrn(pool2)
# Flatten: [-1, 8, 8, 64] -> [-1, 8*8*64 = 4096]
flatten = tf.contrib.layers.flatten(lrn2)
# Fully Connected 1:  [-1, 8*8*64 = 8192] -> [-1, 384]
fc1 = add_fully(flatten, 384)
fc1_drop = tf.nn.dropout(fc1, params['dropout'])
# Output, (fully connected): [-1, 384] -> [-1, 10]
y_predict = add_fully(fc1_drop, 10) # no sofmax, apply softmax in cost funct

# Define cost and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y_predict) )
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Test the accuracy of trained DNN
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_truth,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



''' Define fuctions with use of tf session'''
def train(sess, features, labels, batch_size):
  iterations = int(features.shape[0] / batch_size)
  p = 0
  for i in range(iterations):
    sess.run(train_step, feed_dict={x: features[p:p+batch_size], y_truth: labels[p:p+batch_size]})
    p += batch_size
  return

def calculate_accuracy(sess, features, labels):
  feed_size = 500
  iterations = int(features.shape[0] / feed_size)
  acc = 0.0
  p = 0
  for i in range(iterations):
    acc += sess.run(accuracy, feed_dict={x: features[p:p+feed_size], y_truth: labels[p:p+feed_size]}).tolist()
    p += feed_size
  acc = acc /iterations
  return acc
  
def extract_featuremap(sess, layer, feature, label):
  channels = sess.run( layer, feed_dict={x: image, y_truth: label})[0] 
  # channels contain multi channels of fmap
  # choose one from channels
  which = 0 
  fmap = channels[:,:,which]
  # preserve only four digit after decimal point
  fmap = np.around(fmap, decimals=4)
  tolist = fmap.tolist()
  return tolist

def extract_multiple_featuremaps(sess, feature, label, name):
  log[name]['conv1'] = extract_featuremap(sess, conv1, feature, label)
  log[name]['pool1'] = extract_featuremap(sess, pool1, feature, label)
  log[name]['lrn1'] = extract_featuremap(sess, lrn1, feature, label)
  log[name]['conv2'] = extract_featuremap(sess, conv2, feature, label)
  log[name]['pool2'] = extract_featuremap(sess, pool2, feature, label)
  log[name]['lrn2'] = extract_featuremap(sess, lrn2, feature, label)
  return

# Launch the graph
with tf.Session() as sess:
  # Initializing the variables
  init = tf.global_variables_initializer()
  sess.run(init)
  # Initial accuracy log
  #batch_x, batch_y = train_data.next_batch()

  init_train_acc = calculate_accuracy(sess, train_X, train_Y)
  init_test_acc = calculate_accuracy(sess, test_X, test_Y) #sess.run(accuracy, feed_dict={x: test_batch_x, y_truth: test_batch_y}).tolist()
  log['train_accuracy_per_epoch'].append(init_train_acc)
  log['test_accuracy_per_epoch'].append(init_test_acc)
  # Train DNN for n_epochs times

  for epoch in range(params['epoch']):
    print('------------ Epoch %3d -------------' % (epoch))
    train(sess, train_X, train_Y, params['batch_size'])
    # Output accuracy
    train_acc = calculate_accuracy(sess, train_X[0:501], train_Y[0:501])
    test_acc = calculate_accuracy(sess, test_X[0:501], test_Y[0:501])
    print('train_acc = % .4f \t\t test_acc = %.4f'  %(train_acc, test_acc))
    log['train_accuracy_per_epoch'].append(train_acc)
    log['test_accuracy_per_epoch'].append(test_acc)
    
    # Output feature maps in the middle of training process
    #if epoch == params['epoch'] / 2:
      #extract_multiple_featuremaps(sess, first_image, first_label, name='feature_map_0')
      #log['feature_map_0'] = sess.run( conv1, feed_dict={x: first_image})[0].flatten().tolist() 

  print("Training Finished!")
  print('Predict first image as:', tf.Tensor.eval(tf.argmax( sess.run(y_predict, feed_dict={x: first_image}), 1)[0]))
  print('Groundtruth first image:', tf.Tensor.eval(tf.argmax(first_label, 1)[0]))

  # Output feature maps after training
  #extract_multiple_featuremaps(sess, first_image, first_label, name='feature_map_1')
  #log['feature_map_1'] = sess.run( conv1, feed_dict={x: first_image})[0].flatten().tolist() 


# Merge params data into log
log.update(params)
# Print weights and accuracy log to json file
with open(logfile, 'w') as f:
  json.dump(log, f)
