''' data utility functions '''
import numpy as np

def load_data(filename):
    data = np.load(filename)
    return data

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
