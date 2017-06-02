import numpy as np
import tensorflow as tf
import json
from util import Dataset, load_data
from random import randint
import os

eid = 'normal01_k3'
n_epochs = 100 #200
batch_size = 500
show_steps = 1
discriminator_learning_rate = 0.001
generator_learning_rate = 0.001
reconstruct_learning_rate = 0.001

input_dim = 784
latent_dim = 100
hidden_dim = 512

statistics = {
    'architechture': '3 layers, {}-{}-{}/{}'.format(hidden_dim, hidden_dim, latent_dim, input_dim),
    'reconstruction_loss': [],
    'generator_loss': [],
    'discriminator_loss': [],
    'encoded_feature_vector': [],
    'original_images': [],
    'encoded_images': [],
    'reconstruct_images': [],
}
statistics_file = 'statistics/'+eid
print('id: ', eid)
print('batch size = {:d}'.format(batch_size))
print('number of epochs = {:d}'.format(n_epochs))

# Load data
X_train = load_data('../data/data.npy')  # (2000, 784)
label_train = load_data('../data/label.npy')  # (2000,)
train_dataset = Dataset(X_train, label_train, batch_size)
n_train_samples = X_train.shape[0]
n_iters = int(n_epochs * n_train_samples / batch_size)
print('number of iterations = {:d}'.format(n_iters))


def dense(x, n_in, n_out, activation):
    b_init = tf.constant_initializer(0.1)
    w_stddev = 0.1 #1.0/np.sqrt(n_in)
    w_init = tf.random_normal_initializer(stddev=w_stddev)
    #w_init = tf.contrib.layers.xavier_initializer(uniform=False)
    out = tf.contrib.layers.fully_connected(x, n_out, activation_fn=None, biases_initializer=b_init, weights_initializer=w_init)
    if activation == 'relu':
        out = tf.nn.relu(out)
    elif activation == 'lrelu':
        tf.maximum(out, 0.2*out)
    elif activation == 'sigmoid':
        out = tf.nn.sigmoid(out)
    return out

def encoder(x):
    #input_layer = tf.nn.dropout(x, 0.8)
    h = dense(x, input_dim, hidden_dim, 'relu')
    h = dense(h, hidden_dim, hidden_dim, 'relu')
    h = dense(h, hidden_dim, latent_dim, 'relu')
    return h

def decoder(code):
    h = dense(code, latent_dim, hidden_dim, 'relu')
    h = dense(h, hidden_dim, hidden_dim, 'relu')
    h = dense(h, hidden_dim, input_dim, 'sigmoid')
    return h

def discriminator(samples):
    h = dense(samples, latent_dim,  hidden_dim, 'relu')
    h = dense(h, hidden_dim, hidden_dim, 'relu')
    h = dense(h, hidden_dim, 1, 'sigmoid')
    return h

def draw_gaussian(dimension, n_samples):
    return np.random.standard_normal((n_samples, dimension)).astype('float32')

def draw_multivariate_gaussian(dimension, n_samples):
    means = np.zeros(dimension)
    cov_matrix = np.identity(dimension)
    return np.random.multivariate_normal(means, cov_matrix, n_samples).astype('float32')

# Graph input
x = tf.placeholder('float', [None, 784])
prior_samples = tf.placeholder('float', [batch_size, latent_dim])
# Network outputs
code = encoder(x)
reconstruct = decoder(code)
discrim_prior = discriminator(prior_samples)
discrim_code = discriminator(code)
# Define loss and optimizer
loss_discriminator = tf.negative(tf.reduce_mean(tf.log(discrim_prior+1e-6)) + tf.reduce_mean(tf.ones_like(discrim_code)-discrim_code+1e-6))
loss_encoder = tf.reduce_mean(tf.log(1.0-discrim_code+1e-6))
loss_reconstruct = tf.reduce_sum(tf.abs(x - reconstruct))
#train_step = tf.train.AdamOptimizer(adam_learning_rate).minimize(loss_discriminator+loss_encoder+loss_reconstruct)
train_discriminator = tf.train.AdamOptimizer(discriminator_learning_rate).minimize(loss_discriminator)
train_generator = tf.train.AdamOptimizer(generator_learning_rate).minimize(loss_encoder)
train_reconstruct = tf.train.AdamOptimizer(reconstruct_learning_rate).minimize(loss_reconstruct)


def record_loss(sess, X):
    iterations = int(X.shape[0] / batch_size)
    reconstruction_loss, generator_loss, discriminator_loss = 0.0, 0.0, 0.0
    p = 0
    for i in range(iterations):
        reconstruction_loss += sess.run(loss_reconstruct, feed_dict={x: X[p:p+batch_size]}).tolist()
        generator_loss += sess.run(loss_encoder, feed_dict={x: X[p:p+batch_size]}).tolist()
        draw_prior_samples = draw_multivariate_gaussian(latent_dim, batch_size)
        discriminator_loss += sess.run(loss_discriminator, feed_dict={x: X[p:p+batch_size], prior_samples: draw_prior_samples}).tolist()
        p += batch_size
    # Average
    reconstruction_loss /= X.shape[0] 
    generator_loss /= X.shape[0] 
    discriminator_loss /= X.shape[0]
    # Record
    statistics['reconstruction_loss'].append(reconstruction_loss)
    statistics['generator_loss'].append(generator_loss)
    statistics['discriminator_loss'].append(discriminator_loss)
    print('Loss: reconstruction = {:.5f},  generator = {:.20f},  discriminator = {:.20f}'.format(reconstruction_loss, generator_loss, discriminator_loss))

def extract_encoded_data(sess):
    iterations = int(train_dataset.X.shape[0] / batch_size)
    p = 0
    for i in range(iterations):
        if i == 0:
            encoded_feature_vector = sess.run(code, feed_dict={x: train_dataset.X[p:p+batch_size]})
            print(encoded_feature_vector.shape)
        else:
            encoded_feature_vector = np.append(encoded_feature_vector, sess.run(code, feed_dict={x: train_dataset.X[p:p+batch_size]}), axis=0)
            print(encoded_feature_vector.shape)
        p += batch_size
    label = train_dataset.Y
    statistics['encoded_feature_vector'] = encoded_feature_vector
    
def extract_image(sess):
    num_images = 10
    ori_images = train_dataset.X[0:num_images+1]
    encoded_images = sess.run(code, feed_dict={x: ori_images})
    reconstruct_images = sess.run(reconstruct, feed_dict={x: ori_images})
    statistics['original_images'] = ori_images
    statistics['encoded_images'] = encoded_images
    statistics['reconstruct_images'] = reconstruct_images



saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, 'models/%s/%s.ckpt' % (eid, n_iters))
    record_loss(sess, train_dataset.X)
    k = 3
    n_iters = int(n_iters/k)
    for it in range(n_iters):
        # Shuffle data once for each epoch
        if it % int(n_iters/n_epochs)  == 0:
            train_dataset.shuffle()
        # Train
        for i in range(k):
            next_x, _ = train_dataset.next_batch()
            draw_prior_samples = draw_multivariate_gaussian(latent_dim, batch_size)
            sess.run(train_discriminator, feed_dict={x: next_x, prior_samples: draw_prior_samples})
            sess.run(train_reconstruct, feed_dict={x: next_x})
        sess.run(train_generator, feed_dict={x: next_x})
        # Show loss
        if it % show_steps == 0:
            print('Iterations %5d: ' %(it+1) , end='')
            record_loss(sess, train_dataset.X)
    extract_encoded_data(sess)
    extract_image(sess)
    # Save the model
    if not os.path.exists('models/'+eid):
        os.makedirs('models/'+eid)
    save_path = saver.save(sess, 'models/%s/%s.ckpt' % (eid, n_iters))
    print('Model saved in file: %s' % save_path)


np.savez(statistics_file,
    reconstruction_loss=statistics['reconstruction_loss'], generator_loss=statistics['generator_loss'],
    discriminator_loss=statistics['discriminator_loss'],
    original_images=statistics['original_images'], encoded_images=statistics['encoded_images'], reconstruct_images=statistics['reconstruct_images'],
    encoded_feature_vector=statistics['encoded_feature_vector'], label = train_dataset.Y,
    architechture=statistics['architechture'])
print('statistics file saved in: {}'.format(statistics_file))
