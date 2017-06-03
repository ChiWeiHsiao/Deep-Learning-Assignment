import numpy as np
import tensorflow as tf
import json
from util import Dataset, load_data
from random import randint
import os

eid = 'smallrate-e500-b100'
n_epochs = 500
batch_size = 100#500
show_steps = 1
discriminator_learning_rate = 0.0001
generator_learning_rate = 0.0001
reconstruct_learning_rate = 0.001

input_dim = 784
latent_dim = 100
hidden_dim = 512

statistics = {
    'architechture': '3 layers, {}-{}-{}/{}'.format(hidden_dim, hidden_dim, latent_dim, input_dim),
    'leaning_rate': 'Learning Rate, disc={:f}, gen={:f}, reconst={:f}'.format(discriminator_learning_rate,generator_learning_rate,reconstruct_learning_rate),
    'reconstruction_loss': [],
    'generator_loss': [],
    'discriminator_loss': [],
    'encoded_feature_vector': [],
    'original_images': [],
    'encoded_images': [],
    'reconstruct_images': [],
    'reconstruct_from_random': [],
}
statistics_file = 'statistics/'+eid
print('id: ', eid)
print('number of epochs = {:d}'.format(n_epochs))
print('batch_size = {:d}'.format(batch_size))
print('Learning Rate, disc={:f}, gen={:f}, reconst={:f}'.format(discriminator_learning_rate,generator_learning_rate,reconstruct_learning_rate))

# Load data
X_train = load_data('../data/data.npy')  # (2000, 784)
label_train = load_data('../data/label.npy')  # (2000,)
train_dataset = Dataset(X_train, label_train, batch_size)
n_train_samples = X_train.shape[0]
n_iters = int(n_epochs * n_train_samples / batch_size)
print('number of iterations = {:d}'.format(n_iters))

def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.fill(shape, 0.1)
    return tf.Variable(initial)

weight = {
    'enc_1': weight_variable([input_dim, hidden_dim]),
    'enc_2': weight_variable([hidden_dim, hidden_dim]),
    'enc_3': weight_variable([hidden_dim, latent_dim]),

    'dec_1': weight_variable([latent_dim, hidden_dim]),
    'dec_2': weight_variable([hidden_dim, hidden_dim]),
    'dec_3': weight_variable([hidden_dim, input_dim]),
    
    'dis_1': weight_variable([latent_dim,  hidden_dim]),
    'dis_2': weight_variable([hidden_dim, hidden_dim]),
    'dis_3': weight_variable([hidden_dim, 1]),
}

bias = {
    'enc_1': bias_variable([hidden_dim]),
    'enc_2': bias_variable([hidden_dim]),
    'enc_3': bias_variable([latent_dim]),

    'dec_1': bias_variable([hidden_dim]),
    'dec_2': bias_variable([hidden_dim]),
    'dec_3': bias_variable([input_dim]),
    
    'dis_1': bias_variable([hidden_dim]),
    'dis_2': bias_variable([hidden_dim]),
    'dis_3': bias_variable([1]),
}
    
def dense(x, W, b, activation):
    out = tf.add(tf.matmul(x, W), b)
    out = tf.layers.batch_normalization(out)
    if activation == 'relu':
        out = tf.nn.relu(out)
    elif activation == 'lrelu':
        tf.maximum(out, 0.2*out)
    elif activation == 'sigmoid':
        out = tf.nn.sigmoid(out)
    return out

def encoder(x):
    #input_layer = tf.nn.dropout(x, 0.8)
    h = dense(x, weight['enc_1'], bias['enc_1'], 'relu')
    h = dense(h, weight['enc_2'], bias['enc_2'], 'relu')
    h = dense(h, weight['enc_3'], bias['enc_3'], 'relu')
    return h

def decoder(x):
    h = dense(x, weight['dec_1'], bias['dec_1'], 'relu')
    h = dense(h, weight['dec_2'], bias['dec_2'], 'relu')
    h = dense(h, weight['dec_3'], bias['dec_3'], 'sigmoid')
    return h

def discriminator(x):
    h = dense(x, weight['dis_1'], bias['dis_1'], 'relu')
    h = dense(h, weight['dis_2'], bias['dis_2'], 'relu')
    h = dense(h, weight['dis_3'], bias['dis_3'], 'sigmoid')
    return h

def draw_gaussian(dimension, n_samples):
    return np.random.standard_normal((n_samples, dimension)).astype('float32')

def draw_multivariate_gaussian(dimension, n_samples):
    means = np.zeros(dimension)
    cov_matrix = np.identity(dimension)
    return np.random.multivariate_normal(means, cov_matrix, n_samples).astype('float32')

# Network to train
x = tf.placeholder('float', [None, 784])
prior_samples = tf.placeholder('float', [batch_size, latent_dim])

code = encoder(x)
reconstruct = decoder(code)
discrim_prior = discriminator(prior_samples)
discrim_code = discriminator(code)

loss_discriminator = tf.negative(tf.reduce_mean(tf.log(discrim_prior+1e-9)) + tf.reduce_mean(tf.ones_like(discrim_code)-discrim_code+1e-9))
loss_encoder = tf.reduce_mean(tf.log(1.0-discrim_code+1e-9))
loss_reconstruct = tf.reduce_sum(tf.abs(x - reconstruct))


train_discriminator = tf.train.AdamOptimizer(discriminator_learning_rate).minimize(loss_discriminator)
train_generator = tf.train.AdamOptimizer(generator_learning_rate).minimize(loss_encoder)
train_reconstruct = tf.train.AdamOptimizer(reconstruct_learning_rate).minimize(loss_reconstruct)

# Reconstruct from random distribution with trained weights
specified_code = tf.placeholder('float', [None, 100])
reconstruct_specified_code = decoder(specified_code)


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
# Train the network
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, 'models/%s/%s.ckpt' % (eid, 533))
    record_loss(sess, train_dataset.X)
    k = 3
    new_iters = int(n_iters/k)
    for it in range(new_iters):
        # Shuffle data once for each epoch
        if it % int(new_iters/n_epochs)  == 0:
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

# Reconstruct with the same weights
with tf.Session() as sess:
    print('Model restore from: %s' % save_path)
    saver.restore(sess, save_path)#'models/%s/%s.ckpt' % (eid, 533))
    draw_prior_samples = draw_multivariate_gaussian(latent_dim, batch_size)
    statistics['reconstruct_from_random'] = sess.run(reconstruct_specified_code, feed_dict={specified_code: draw_prior_samples})


np.savez(statistics_file,
    reconstruction_loss=statistics['reconstruction_loss'], generator_loss=statistics['generator_loss'],
    discriminator_loss=statistics['discriminator_loss'],
    original_images=statistics['original_images'], encoded_images=statistics['encoded_images'], reconstruct_images=statistics['reconstruct_images'],
    encoded_feature_vector=statistics['encoded_feature_vector'], label = train_dataset.Y,
    random_reconstruct_img=statistics['reconstruct_from_random'],
    architechture=statistics['architechture'], leaning_rate=statistics['leaning_rate'])
print('statistics file saved in: {}'.format(statistics_file))
