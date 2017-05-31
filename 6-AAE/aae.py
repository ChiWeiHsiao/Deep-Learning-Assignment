import numpy as np
import tensorflow as tf
import json
from util import Dataset, load_data
from random import randint

eid = 'duplicate-pool'
n_epochs = 5
batch_size = 32
show_steps = 100
adam_learning_rate = 0.001
log = {
    'experiment_id': eid,
    'train_loss': [],
    'encoded_data': [],
}
logfile = 'statistics/'+eid+'.json'
print('id: ', eid)
print('batch size = {:d}'.format(batch_size))
print('number of epochs = {:d}'.format(n_epochs))

# Load data
X_train = load_data('../data/data.npy')  # (2000, 784)
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = X_train.shape[1:4]
n_train_samples = X_train.shape[0]
n_iters = int(n_epochs * n_train_samples / batch_size)
print('number of iterations = {:d}'.format(n_iters))
train_dataset = Dataset(X_train, batch_size)


def dense(x, n_out):
    w_init = tf.random_normal_initializer 
    b_init = tf.random_normal_initializer
    return tf.contrib.layers.fully_connected(x, n_out, activation_fn=None, biases_initializer=b_init, weights_initializer=w_init)

def encoder(x):
    e_dense1 = dense(x, )
    return e_dense1

def decoder(code):
    d_dense1 = dense(code, )
    reconstruct = d_dense1
    return reconstruct

def discriminator(sample, encoded):
    sigmoid 

# Graph input
x = tf.placeholder('float', [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
out = 

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
    log['train_loss'].append(train_loss)
    log['test_loss'].append(test_loss)
    print('train_loss = %15.4f, test_loss = %15.4f'  %(train_loss, test_loss))
    return


def extract_encoded_data(sess):
    feature_vector = sess.run(out, feed_dict={x: }).tolist()
    log['encoded_data'].append(feature_vector)
    

saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, 'models/try.ckpt')
    extract_encoded_data(sess)
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
    save_path = saver.save(sess, 'models/%s.ckpt' % eid)
    print('Model saved in file: %s' % save_path)


# Print weights and accuracy log to json file
with open(logfile, 'w') as f:
    json.dump(log, f, indent=1)
