from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical, Dataset

architecture = 'LSTM'
optimizer = 'ADAM' # 'ADAM' or 'SGD'
eid = 'tf_' + architecture + '_' + optimizer
n_epochs = 4
batch_size = 32
show_steps = 50 # show statistics after train 50 batches
init_learning_rate = 0.2
n_hidden = 10

log = {
  'experiment_id': eid,
  'train_accuracy': [],
  'test_accuracy': [],
  'train_loss': [],
  'test_loss': [],
  'n_hidden': n_hidden,
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
n_samples = X_train.shape[0]
n_time_steps = X_train.shape[1]
n_input = X_train.shape[2]
n_classes = 2
n_iters = int(n_epochs * n_samples / batch_size)
# transform to one-hot
Y_train = to_categorical(Y_train, 2)
Y_test = to_categorical(Y_test, 2)
# Convert to Dataset instance 
train_dataset = Dataset(X_train, Y_train, batch_size)


def RNN(x_sequence, n_hidden):
  cell = rnn.BasicRNNCell(n_hidden)
  outputs, states = rnn.static_rnn(cell, x_sequence, dtype=tf.float32)
  # use the last output of rnn cell to compute cost function
  weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
  bias = tf.Variable(tf.random_normal([n_classes]))
  return tf.matmul(outputs[-1], weight) + bias
  
def LSTM(x_sequence, n_hidden):
  cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
  outputs, states = rnn.static_rnn(cell, x_sequence, dtype=tf.float32)
  # use the last output of rnn cell to compute cost function
  weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
  bias = tf.Variable(tf.random_normal([n_classes]))
  return tf.matmul(outputs[-1], weight) + bias

def GRU(x_sequence, n_hidden):
  cell = rnn.GRUCell(n_hidden)
  outputs, states = rnn.static_rnn(cell, x_sequence, dtype=tf.float32)
  # use the last output of rnn cell to compute cost function
  weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
  bias = tf.Variable(tf.random_normal([n_classes]))
  return tf.matmul(outputs[-1], weight) + bias


#### Define RNN model ####
# Graph input
x = tf.placeholder('float', [None, n_time_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

# Unstack to get a list of 'n_time_steps' tensors of shape (batch_size, n_input)
x_sequence = tf.unstack(x, n_time_steps, 1)
if(architecture == 'RNN'):
  predict = RNN(x_sequence, n_hidden)
elif(architecture == 'LSTM'):
  predict = LSTM(x_sequence, n_hidden)
elif(architecture == 'GRU'):
  predict = GRU(x_sequence, n_hidden)

# Define cost and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict) )

if optimizer == 'ADAM':
  adam_learning_rate = 0.001
  train_step = tf.train.AdamOptimizer(adam_learning_rate).minimize(cost)
# SGD with Exponentially decayed learning rate
elif optimizer == 'SGD':
  batch = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(init_learning_rate, batch*batch_size, 1000, 0.95, staircase=True)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=batch)

# Test the accuracy of trained DNN
correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#### Define fuctions with use of tf session ####
def calculate_accuracy(sess, features, labels):
  feed_size = 500
  iterations = int(features.shape[0] / feed_size)
  acc = 0.0
  p = 0
  for i in range(iterations):
    acc += sess.run(accuracy, feed_dict={x: features[p:p+feed_size], y: labels[p:p+feed_size]}).tolist()
    p += feed_size
  acc = acc /iterations
  return acc

def calculate_loss(sess, features, labels):
  feed_size = 500
  iterations = int(features.shape[0] / feed_size)
  loss = 0.0
  p = 0
  for i in range(iterations):
    loss += sess.run(cost, feed_dict={x: features[p:p+feed_size], y: labels[p:p+feed_size]}).tolist()
    p += feed_size
  return loss

def record_accuracy_and_loss(sess):
  train_acc = calculate_accuracy(sess, X_train, Y_train)
  test_acc = calculate_accuracy(sess, X_test, Y_test)
  log['train_accuracy'].append(train_acc)
  log['test_accuracy'].append(test_acc)
  print('train_acc = % .4f, test_acc = %.4f, '  %(train_acc, test_acc), end='')
  train_loss = calculate_loss(sess, X_train, Y_train)
  test_loss = calculate_loss(sess, X_test, Y_test)
  log['train_loss'].append(train_loss)
  log['test_loss'].append(test_loss)
  print('train_loss = % .4f, test_loss = %.4f'  %(train_loss, test_loss))
  return
  

saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  # Restore model
  #saver.restore(sess, 'models/try_2.ckpt')
  #print("Model restored.")
  print('Before train:\t', end="")
  record_accuracy_and_loss(sess)
  for it in range(n_iters):
    # Train next batch
    next_x, next_y = train_dataset.next_batch()
    sess.run(train_step, feed_dict={x: next_x, y: next_y})
    # Record accuracy and loss
    if it % show_steps == 0:
      print('Iterations %4d:\t' %(it+1) , end="")
      record_accuracy_and_loss(sess)
    # Shuffle data once for each epoch
    if it % int(n_iters/n_epochs) == 0:
      train_dataset.shuffle()
    
  # Save the model
  save_path = saver.save(sess, 'models/%s.ckpt' % eid)
  print('Model saved in file: %s' % save_path)


# Print weights and accuracy log to json file
with open(logfile, 'w') as f:
  json.dump(log, f, indent=1)
