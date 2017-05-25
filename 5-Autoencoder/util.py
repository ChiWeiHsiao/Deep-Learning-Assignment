''' data utility functions '''
import numpy as np

def load_data():
    MNIST_M = np.load('../data/Mnist_M.npy')
    train_data = MNIST_M[0][0] # 59000
    valid_data = MNIST_M[1][0] # 1000
    test_data = MNIST_M[2][0] # 10000
    return train_data, valid_data, test_data

class Dataset():
  def __init__(self, X, batch_size):
    self.X = X
    self.shuffle()
    self.batch_size = batch_size
    self.state = 0
    self.total_batch = int(X.shape[0] / batch_size)

  def next_batch(self):
    start = self.state * self.batch_size
    end = start + self.batch_size
    next_x = self.X[start:end]
    self.state += 1
    self.state %= self.total_batch
    return next_x

  def shuffle(self):
    np.random.shuffle(self.X)
