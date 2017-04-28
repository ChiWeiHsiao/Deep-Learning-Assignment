from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical, Dataset

architecture = 'GRU'
eid = 'handcraft_' + architecture + '_1'
n_epochs = 5
batch_size = 32
show_steps = 50 # show statistics after train 50 batches
learning_rate = 0.001
n_hidden = 2 # number of hidden nodes

log = {
  'experiment_id': eid,
  'train_accuracy': [],
  'test_accuracy': [],
  'train_loss': [],
  'test_loss': [],
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
n_input = X_train.shape[2]
n_classes = 2
n_iters = int(n_epochs * n_sequences / batch_size)
# transform to one-hot
Y_train = to_categorical(Y_train, 2)
Y_test = to_categorical(Y_test, 2)
# Convert to Dataset instance 
train_dataset = Dataset(X_train, Y_train, batch_size)


#### Define RNN Models ####
def RNN(x_sequence, n_hidden):
  state = tf.Variable(tf.zeros([batch_size, n_hidden]))
  U = tf.Variable(tf.random_normal([n_input, n_hidden]))
  W = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
  hidden_bias = tf.Variable(tf.random_normal([batch_size, n_hidden]))
  # Unroll timesteps
  for x in x_sequence:
    state = tf.matmul(x, U) + tf.matmul(state, W) + hidden_bias
    # (batch, n_input)x(n_input, n_hidden) + (batch_size, n_hidden)x(n_hidden, n_hidden) + (batch_size, n_hidden)
  return state


def LSTM(x_sequence, n_hidden):
  # Parameters
  # forgate gate
  fb = tf.Variable(tf.random_normal([batch_size, n_hidden]))
  fU = tf.Variable(tf.random_normal([n_input, n_hidden]))
  fW = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
  # internal state
  sb = tf.Variable(tf.random_normal([batch_size, n_hidden]))
  sU = tf.Variable(tf.random_normal([n_input, n_hidden]))
  sW = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
  # external input gate
  ib = tf.Variable(tf.random_normal([batch_size, n_hidden]))
  iU = tf.Variable(tf.random_normal([n_input, n_hidden]))
  iW = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
  # output gate
  ob = tf.Variable(tf.random_normal([batch_size, n_hidden]))
  oU = tf.Variable(tf.random_normal([n_input, n_hidden]))
  oW = tf.Variable(tf.random_normal([n_hidden, n_hidden]))

  def lstm_cell(x, output, state):
    # (batch, n_input)x(n_input, n_hidden) + (batch_size, n_hidden)x(n_hidden, n_hidden) + (batch_size, n_hidden)
    forget_gate = tf.sigmoid(tf.matmul(x, fU) + tf.matmul(output, fW) + fb)
    input_gate = tf.sigmoid(tf.matmul(x, iU) + tf.matmul(output, iW) + ib)
    state_update = tf.sigmoid(tf.matmul(x, sU) + tf.matmul(output, sW) + sb)
    state = state * forget_gate + input_gate * state_update
    output_gate = tf.sigmoid(tf.matmul(x, oU) + tf.matmul(output, oW) + ob)
    output = tf.tanh(state) * output_gate
    return output, state 
  
  # Unroll timesteps
  state = tf.Variable(tf.zeros([batch_size, n_hidden]))
  output = tf.Variable(tf.zeros([batch_size, n_hidden]))
  for x in x_sequence:
    output, state = lstm_cell(x, output, state) 
  return output

def GRU(x_sequence, n_hidden):
  # Parameters
  # update gate
  ub = tf.Variable(tf.random_normal([batch_size, n_hidden]))
  uU = tf.Variable(tf.random_normal([n_input, n_hidden]))
  uW = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
  # reset state
  rb = tf.Variable(tf.random_normal([batch_size, n_hidden]))
  rU = tf.Variable(tf.random_normal([n_input, n_hidden]))
  rW = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
  # state
  b = tf.Variable(tf.random_normal([batch_size, n_hidden]))
  U = tf.Variable(tf.random_normal([n_input, n_hidden]))
  W = tf.Variable(tf.random_normal([n_hidden, n_hidden]))

  def gru_cell(x, state):
    update_gate = tf.sigmoid(tf.matmul(x, uU) + tf.matmul(state, uW) + ub)
    reset_gate = tf.sigmoid(tf.matmul(x, rU) + tf.matmul(state, rW) + rb)
    state_update = tf.sigmoid(tf.matmul(x, U) + tf.matmul(reset_gate, W) * state + b)
    state = update_gate * state + (1-update_gate) * state_update
    return state 
  
  # Unroll timesteps
  state = tf.Variable(tf.zeros([batch_size, n_hidden]))
  for x in x_sequence:
    state = gru_cell(x, state) 
  return state



#### Define whole model ####
# Graph input
x = tf.placeholder('float', [batch_size, n_time_steps, n_input])
y = tf.placeholder('float', [batch_size, n_classes])

# Unstack to get a list of 'n_time_steps' tensors of shape (batch_size, n_input)
x_sequence = tf.unstack(x, n_time_steps, 1)
if architecture == 'RNN':
  rnn_output = RNN(x_sequence, n_hidden)
elif architecture == 'LSTM':
  rnn_output = LSTM(x_sequence, n_hidden)
elif architecture == 'GRU':
  rnn_output = GRU(x_sequence, n_hidden)

# add a fully connected layer without activation after rnn
weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([batch_size, n_classes]))
predict = tf.matmul(rnn_output, weight) + bias

# Define cost and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict) )
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Test the accuracy of trained DNN
correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#### Define fuctions with use of tf session ####
def calculate_accuracy(sess, features, labels):
  iterations = int(features.shape[0] / batch_size)
  acc = 0.0
  p = 0
  for i in range(iterations):
    acc += sess.run(accuracy, feed_dict={x: features[p:p+batch_size], y: labels[p:p+batch_size]}).tolist()
    p += batch_size
  acc = acc /iterations
  return acc

def calculate_loss(sess, features, labels):
  iterations = int(features.shape[0] / batch_size)
  loss = 0.0
  p = 0
  for i in range(iterations):
    loss += sess.run(cost, feed_dict={x: features[p:p+batch_size], y: labels[p:p+batch_size]}).tolist()
    p += batch_size
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
    if(it % show_steps == 0):
      print('Iterations %4d:\t' %(it+1) , end="")
      record_accuracy_and_loss(sess)
    # Shuffle data once for each epoch
    if(it % batch_size == 0):
      train_dataset.shuffle()
    
  # Save the model
  save_path = saver.save(sess, 'models/%s.ckpt' % eid)
  print('Model saved in file: %s' % save_path)


# Print weights and accuracy log to json file
with open(logfile, 'w') as f:
  json.dump(log, f, indent=1)
