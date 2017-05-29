import numpy as np
import tensorflow as tf
import json
from util import Dataset, load_data
from random import randint

eid = 'nopool'
n_epochs = 5
batch_size = 32
show_steps = 100
adam_learning_rate = 0.001
log = {
  'experiment_id': eid,
  'train_loss': [],
  'test_loss': [],
  'original_image': [],
  'reconstruct_image': [],
}
logfile = 'statistics/'+eid+'.json'
print('id: ', eid)
print('batch size = {:d}'.format(batch_size))
print('number of epochs = {:d}'.format(n_epochs))

# Load data
X_train, X_valid, X_test = load_data() # X_train = (59000, 28, 28, 3)
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = X_train.shape[1:4]
n_train_samples = X_train.shape[0]
n_iters = int(n_epochs * n_train_samples / batch_size)
print('number of iterations = {:d}'.format(n_iters))
train_dataset = Dataset(X_train, batch_size)


def conv2d(x, out_channels, kernel, stride):
  return tf.contrib.layers.conv2d(x, out_channels, kernel, stride, activation_fn=tf.nn.relu, 
      biases_initializer=tf.constant_initializer(0.), weights_initializer=tf.random_normal_initializer, padding='SAME')

def conv2d_transpose(x, out_channels, kernel, stride):
  return tf.contrib.layers.conv2d_transpose(x, out_channels, kernel, stride, activation_fn=tf.nn.relu, 
      biases_initializer=tf.constant_initializer(0.), weights_initializer=tf.random_normal_initializer, padding='SAME')

def maxpool_2x2(x):
    return tf.contrib.layers.max_pool2d(x, kernel_size=2, stride=2, padding='SAME') 
    #tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', Targmax=tf.int32)

def unpool_2x2_duplicate(x, out_shape):
    duplicate = tf.concat([x, x], 3)
    duplicate = tf.concat([duplicate, duplicate], 2)
    out = tf.reshape(duplicate, out_shape)
    return out

def unpool_2x2_nearest_neighbor(x, out_shape):
    return tf.image.resize_nearest_neighbor(x, tf.stack([out_shape[1], out_shape[2]]))


# Graph input
x = tf.placeholder('float', [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
# Encoding pass
encode_conv1 = conv2d(x, out_channels=16, kernel=3, stride=1)
encode_conv2 = conv2d(encode_conv1, out_channels=32, kernel=3, stride=1)
code_layer = encode_conv2
# Decoding pass
decode_conv2 = conv2d_transpose(code_layer, out_channels=16, kernel=3, stride=1)
decode_conv1 = conv2d_transpose(decode_conv2, out_channels=3, kernel=3, stride=1)
out = decode_conv1
#print("unconv layer: {}".format(decode_conv1.get_shape()))

# Define cost and optimizer
cost = tf.reduce_mean(tf.squared_difference(x, out))
train_step = tf.train.AdamOptimizer(adam_learning_rate).minimize(cost)


def calculate_loss_sum(sess, inputs):
  iterations = int(inputs.shape[0] / batch_size)
  loss = 0.0
  p = 0
  for i in range(iterations):
    loss += sess.run(cost, feed_dict={x: inputs[p:p+batch_size]}).tolist()
    p += batch_size
  return loss

def record_loss(sess):
  train_loss = calculate_loss_sum(sess, X_train) / X_train.shape[0]
  test_loss = calculate_loss_sum(sess, X_test) / X_test.shape[0]
  log['train_loss'].append(train_loss)
  log['test_loss'].append(test_loss)
  print('train_loss = %15.4f, test_loss = %15.4f'  %(train_loss, test_loss))
  return


def extract_images(sess):
  for i in range(0,10):
    image_id = randint(0, X_test.shape[0]-1)
    image = X_test[image_id]
    image = np.reshape(image, (1, )+image.shape)
    log['original_image'].append(image.tolist())
    log['reconstruct_image'].append(sess.run( out, feed_dict={x: image}).tolist())
  return
  

saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  #saver.restore(sess, 'models/try.ckpt')
  extract_images(sess)
  record_loss(sess)
  for it in range(n_iters):
    # Train next batch
    next_x = train_dataset.next_batch()
    sess.run(train_step, feed_dict={x: next_x})
    if it % show_steps == 0:
      print('Iterations %4d:\t' %(it+1) , end="")
      record_loss(sess)
    # Shuffle data once for each epoch
    if it % int(n_iters/n_epochs)  == 0:
      train_dataset.shuffle()
    
  # Save the model
  #save_path = saver.save(sess, 'models/%s.ckpt' % eid)
  #print('Model saved in file: %s' % save_path)


# Print weights and accuracy log to json file
with open(logfile, 'w') as f:
  json.dump(log, f, indent=1)
