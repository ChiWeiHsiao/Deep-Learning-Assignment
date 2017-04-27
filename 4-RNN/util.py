''' data utility functions '''
import numpy as np

def shuffle(*pairs):
  pairs = list(pairs)
  for i, pair in enumerate(pairs):
    pairs[i] = np.array(pair)
  p = np.random.permutation(len(pairs[0]))
  return tuple(pair[p] for pair in pairs)

def to_categorical(y, nb_classes):
  y = np.asarray(y, dtype='int32')
  Y = np.zeros((len(y), nb_classes))
  Y[np.arange(len(y)),y] = 1.
  return Y
