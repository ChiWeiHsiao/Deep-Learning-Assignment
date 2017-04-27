from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical

eid = 'try'
n_epochs = 3
batch_size = 20
learning_rate = 0.001
n_hiddens = 2

log = {
  'experiment_id': eid,
  'train_accuracy_per_epoch': [],
  'test_accuracy_per_epoch': [],
}
logfile = 'statistics/'+eid+'.json'
print('id: ', eid)
print('num of epochs:', n_epochs)


#### Load imdb data ####
imdb = np.load('../data/imdb_word_emb.npz')
X_train = imdb['X_train'] #(25000, 80, 128)
Y_train = imdb['y_train'] #(25000, ) => 0 or 1 
X_test  = imdb['X_test']
Y_test  = imdb['y_test']
n_sequences = X_train.shape[0]
n_time_steps = X_train.shape[1]
input_size = X_train.shape[2]
n_classes = 2
# transform to one-hot
Y_train = to_categorical(Y_train, 2)
Y_test = to_categorical(Y_test, 2)


#### Define RNN model ####
# Graph input
x = tf.placeholder('float', [None, n_time_steps, input_size])
y = tf.placeholder('float', [None, n_classes])

# Unstack to get a list of 'n_time_steps' tensors of shape (batch_size, input_size)
x_time_steps = tf.unstack(x, n_time_steps, 1)
rnn_cell = rnn.BasicRNNCell(n_hiddens)
outputs, states = rnn.static_rnn(rnn_cell, x_time_steps, dtype=tf.float32)
# use the last output of rnn cell to compute cost function
predict = outputs[-1]

# Define cost and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict) )
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Test the accuracy of trained DNN
correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#### Define fuctions with use of tf session ####
def train(sess, features, labels, batch_size):
  iterations = int(n_sequences / batch_size)
  p = 0
  for i in range(iterations):
    sess.run(train_step, feed_dict={x: features[p:p+batch_size], y: labels[p:p+batch_size]})
    p += batch_size
  return

def calculate_accuracy(sess, features, labels):
  feed_size = 500
  iterations = int(features.shape[0] / feed_size)
  acc = 0.0
  p = 0
  for i in range(iterations):
    #tuple()?
    acc += sess.run(accuracy, feed_dict={x: features[p:p+feed_size], y: labels[p:p+feed_size]}).tolist()
    #acc += sess.run(accuracy, feed_dict={x: features[p:p+feed_size], y: labels[p:p+feed_size]}).tolist()
    p += feed_size
  acc = acc /iterations
  return acc

def record_accuracy(sess):
  train_acc = calculate_accuracy(sess, X_train, Y_train)
  test_acc = calculate_accuracy(sess, X_test, Y_test)
  log['train_accuracy_per_epoch'].append(train_acc)
  log['test_accuracy_per_epoch'].append(test_acc)
  print('train_acc = % .4f \t\t test_acc = %.4f'  %(train_acc, test_acc))
  return

  
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  record_accuracy(sess)
  for epoch in range(n_epochs):
    print('Epoch %3d:\t' %(epoch+1) , end="")
    train(sess, X_train, Y_train, batch_size)
    record_accuracy(sess)
  # Save the model
  save_path = saver.save(sess, 'models/%s.ckpt' % eid)
  print('Model saved in file: %s' % save_path)


# Print weights and accuracy log to json file
with open(logfile, 'w') as f:
  json.dump(log, f, indent=1)
