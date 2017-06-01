import numpy as np
import tensorflow as tf
import json
from util import Dataset, load_data
from random import randint
import os

eid = 'try'
n_epochs = 200
batch_size = 500
show_steps = 1
adam_learning_rate = 0.001
statistic = {
    'reconstruction_loss': [],
    'generator_loss': [],
    'discriminator_loss': [],
}
statistic_file = 'statistics/'+eid+'.json'
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


def dense(x, n_out, activation):
    if activation == 'relu':
        activation = tf.nn.relu
    elif activation == 'sigmoid':
        activation = tf.sigmoid
    w_init = tf.truncated_normal_initializer(stddev = 1)
    #w_init = tf.random_normal_initializer(stddev=1)
    #w_init = tf.contrib.layers.xavier_initializer(uniform=False)
    b_init = tf.constant_initializer(0.1)
    return tf.contrib.layers.fully_connected(x, n_out, activation_fn=activation, biases_initializer=b_init, weights_initializer=w_init)

input_dim = 784
latent_dim = 100
hidden_dim = 512

def encoder(x):
    encode = dense(x, hidden_dim, 'relu')
    encode = dense(x, hidden_dim, 'relu')
    encode = dense(encode, latent_dim, 'relu')
    return encode

def decoder(code):
    decode = dense(code, hidden_dim, 'relu')
    decode = dense(code, hidden_dim, 'relu')
    decode = dense(decode, input_dim, 'sigmoid')
    return decode

def discriminator(samples):
    d = dense(samples, hidden_dim, 'relu')
    d = dense(samples, hidden_dim, 'relu')
    d = dense(d, hidden_dim, 'sigmoid')
    return d

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
#loss_discriminator = tf.negative(tf.reduce_mean(tf.log(discrim_prior+1e-6))) + tf.reduce_mean(tf.ones_like(discrim_code)-discrim_code)
#loss_encoder = tf.reduce_mean(tf.log(1.0-discrim_code+1e-6))
loss_discriminator = tf.negative(tf.reduce_mean(tf.log(discrim_prior+1e-6))) + tf.reduce_mean(tf.ones_like(discrim_code)-discrim_code+1e-6)
loss_encoder = tf.reduce_mean(tf.log(1.0-discrim_code+1e-6))
loss_reconstruct = tf.reduce_sum(tf.abs(x - reconstruct))
#train_step = tf.train.AdamOptimizer(adam_learning_rate).minimize(loss_discriminator+loss_encoder+loss_reconstruct)
train_discriminator = tf.train.AdamOptimizer(adam_learning_rate).minimize(loss_discriminator)
train_generator = tf.train.AdamOptimizer(adam_learning_rate).minimize(loss_encoder)
train_reconstruct = tf.train.AdamOptimizer(adam_learning_rate).minimize(loss_reconstruct)


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
    statistic['reconstruction_loss'].append(reconstruction_loss)
    statistic['generator_loss'].append(generator_loss)
    statistic['discriminator_loss'].append(discriminator_loss)
    print('Loss: reconstruction = {:.4f},  generator = {:.10f},  discriminator = {:.10f}'.format(reconstruction_loss, generator_loss, discriminator_loss))

def extract_encoded_data(sess, outfile):
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
    np.savez(outfile, feature_vector=encoded_feature_vector, label=label)
    print('Encoded feature vector saved in: %s' %outfile)
    
def extract_image(sess, outfile):
    num_images = 10
    ori_images = train_dataset.X[0:num_images+1]
    encoded_images = sess.run(code, feed_dict={x: ori_images})
    reconstruct_images = sess.run(reconstruct, feed_dict={x: ori_images})
    np.savez(outfile, original_images=ori_images, encoded_images=encoded_images, reconstruct_images=reconstruct_images)
    print('Original / encoded / reconstruct images saved in: %s' %outfile)



saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, 'models/%s/%s.ckpt' % (eid, n_iters))
    record_loss(sess, train_dataset.X)
    for it in range(n_iters):
        # Shuffle data once for each epoch
        if it % int(n_iters/n_epochs)  == 0:
            train_dataset.shuffle()
        # Train next batch
        next_x, _ = train_dataset.next_batch()
        draw_prior_samples = draw_multivariate_gaussian(latent_dim, batch_size)
        sess.run(train_discriminator, feed_dict={x: next_x, prior_samples: draw_prior_samples})
        sess.run(train_generator, feed_dict={x: next_x})
        sess.run(train_reconstruct, feed_dict={x: next_x})
        # Show loss
        if it % show_steps == 0:
            print('Iterations %5d: ' %(it+1) , end='')
            record_loss(sess, train_dataset.X)

    extract_encoded_data(sess, 'statistics/%s-encoded-data' %eid)
    extract_image(sess, 'statistics/%s-images' %eid)

    # Save the model
    if not os.path.exists('models/'+eid):
        os.makedirs('models/'+eid)
    save_path = saver.save(sess, 'models/%s/%s.ckpt' % (eid, n_iters))
    print('Model saved in file: %s' % save_path)

with open(statistic_file, 'w') as f:
    json.dump(statistic, f, indent=1)
    print('Loss statistics saved in: %s' %statistic_file)
