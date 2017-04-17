from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import json

# Import mnist dataset
mnist = input_data.read_data_sets("/tmp//mnist", one_hot=True)

# Regularization option {None, 'L1', 'L2', 'dropout'}
regularizer = 'L2'
run_id = '5'
experiment_id = str(regularizer)+ '_' + run_id
print(experiment_id)
regular_scale = 0.1 #if use 'L1' or 'L2'
dropout_p = 0.75 #if use dropout
logfile = 'statistics/'+experiment_id
logfile += '.json'

# Training Parameters
n_epochs = 30
batch_size = 128
learning_rate = 0.001

# Network Parameters
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128 
n_hidden_3 = 64
n_hidden_4 = 32
n_out = 10

# Log
log = {
  'experiment_id': experiment_id,
  'train_accuracy_per_epoch': [],
  'test_accuracy_per_epoch': [],
  'weight_1': [],
  'weight_2': [],
  'weight_3': [],
  'weight_4': [],
  'weight_5': [],
  'bias_1': [],
  'bias_2': [],
  'bias_3': [],
  'bias_4': [],
  'bias_5': [],
  'count_zeros': 0,
  'num_total_weight': 0
}

def add_regularization(cost, weights, option='L2', scale=0.0):
  regularization = 0
  if option=='L2':
    for w in weights:
      regularization += tf.contrib.layers.l2_regularizer(scale)(w)
    # multiply by 2, because tf implemntation l2 norm as: sum(t ** 2) / 2'
    regularization = tf.scalar_mul(2, regularization) 
  elif option=='L1':
    for w in weights:
      regularization += tf.contrib.layers.l1_regularizer(scale)(w) #tf implement: sclale * abs(w)
  return tf.reduce_sum(cost+regularization) 


''' Build Computation Gragh  DNN model '''
# Graph input
x = tf.placeholder("float", [None, n_input])
y_truth = tf.placeholder("float", [None, n_out])
# layer 1
w_1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
b_1 = tf.Variable(tf.random_normal([n_hidden_1]))
layer_1 = tf.add(tf.matmul(x, w_1), b_1)
layer_1 = tf.nn.relu(layer_1)
if regularizer == 'dropout':
  layer_1 = tf.nn.dropout(layer_1, dropout_p)
# layer 2
w_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
b_2 = tf.Variable(tf.random_normal([n_hidden_2]))
layer_2 = tf.add(tf.matmul(layer_1, w_2), b_2)
layer_2 = tf.nn.relu(layer_2)
if regularizer == 'dropout':
  layer_2 = tf.nn.dropout(layer_2, dropout_p)
# layer 3
w_3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]))
b_3 = tf.Variable(tf.random_normal([n_hidden_3]))
layer_3 = tf.add(tf.matmul(layer_2, w_3), b_3)
layer_3 = tf.nn.relu(layer_3)
if regularizer == 'dropout':
  layer_3 = tf.nn.dropout(layer_3, dropout_p)
# layer 4
w_4 = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]))
b_4 = tf.Variable(tf.random_normal([n_hidden_4]))
layer_4 = tf.add(tf.matmul(layer_3, w_4), b_4)
layer_4 = tf.nn.relu(layer_4)
if regularizer == 'dropout':
  layer_4 = tf.nn.dropout(layer_4, dropout_p)
# layer 5:  output
w_5 = tf.Variable(tf.random_normal([n_hidden_4, n_out]))
b_5 = tf.Variable(tf.random_normal([n_out]))
out_layer = tf.add(tf.matmul(layer_4, w_5), b_5)
y_predict = out_layer
# Cost Function
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y_predict))
if(regularizer=='L1' or regularizer=='L2'):
  cost = add_regularization(cost, [w_1, w_2, w_3, w_4, w_5], regularizer, regular_scale)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


''' Launch the Graph '''
# Test the accuracy of trained DNN
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_truth,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the graph
with tf.Session() as sess:
  # Initializing the variables
  sess.run(tf.global_variables_initializer())
  # Initial accuracy log
  train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y_truth: mnist.train.labels}).tolist()
  test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_truth: mnist.test.labels}).tolist()
  log['train_accuracy_per_epoch'].append(train_acc)
  log['test_accuracy_per_epoch'].append(test_acc)
  # Train DNN  n_epochs times
  for epoch in range(n_epochs):
    n_batch_in_one_epoch = int(mnist.train.num_examples/batch_size)
    
    for i in range(n_batch_in_one_epoch):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict={x: batch_x, y_truth: batch_y})
    # Output accuracy log per epoch
    train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y_truth: mnist.train.labels}).tolist()
    test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_truth: mnist.test.labels}).tolist()
    log['train_accuracy_per_epoch'].append(train_acc)
    log['test_accuracy_per_epoch'].append(test_acc)
    print("Epoch:", '%03d' % (epoch+1), "train_accuracy=", "{:.9f}".format(train_acc), "test_accuracy=", "{:.9f}".format(test_acc))
    
  # weights and bias, transform ndarray to 1D list
  log['weight_1'] = sess.run(w_1).flatten().tolist()
  log['weight_2'] = sess.run(w_2).flatten().tolist()
  log['weight_3'] = sess.run(w_3).flatten().tolist()
  log['weight_4'] = sess.run(w_4).flatten().tolist()
  log['weight_5'] = sess.run(w_5).flatten().tolist()
  log['bias_1'] = sess.run(b_1).flatten().tolist()
  log['bias_2'] = sess.run(b_2).flatten().tolist()
  log['bias_3'] = sess.run(b_3).flatten().tolist()
  log['bias_4'] = sess.run(b_4).flatten().tolist()
  log['bias_5'] = sess.run(b_5).flatten().tolist()

  weights = log['weight_1'] + log['weight_2'] + log['weight_3'] + log['weight_4'] + log['weight_5']
  log['num_total_weight'] = len(weights)
  log['count_zeros'] =  log['num_total_weight'] - sess.run( tf.count_nonzero(weights)).tolist()

print("Training Finished!")
print('Zero elements in weights: ', log['count_zeros'])
print('Total elements in weights: ', log['num_total_weight'])

params = {
  'regularizer': regularizer,
  'experiment_id': experiment_id,
  'regular_scale': regular_scale,  #if use 'L1' or 'L2'
  'dropout_p': dropout_p, #if use dropout
  'n_epochs': n_epochs,
  'batch_size': batch_size,
  'learning_rate': learning_rate
}
log.update(params)

# Print weights and accuracy log to json file
with open(logfile, 'w') as f:
  json.dump(log, f, indent=1)

