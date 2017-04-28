import matplotlib.pyplot as plt
import json
import numpy as np

id = 'tf_LSTM_1'
filename = 'statistics/'+id
filename += '.json'

with open(filename, 'r') as f:
  log = json.load(f)
  train_accuracy =  log['train_accuracy']
  test_accuracy = log['test_accuracy']
  train_loss =  log['train_loss']
  test_loss = log['test_loss']

# convert accuracy to miss classification rate
for i in range(len(train_accuracy)):
  train_accuracy[i] = 1 - train_accuracy[i]
for i in range(len(test_accuracy)):
  test_accuracy[i] = 1 - test_accuracy[i]

# Show Learning Curve
def plot_curve(typ, train_statistics, test_statistics):
  # find out border
  left = 0
  right = len(train_statistics)
  top = np.amax( np.array(train_statistics) )
  bottom = 0

  plt.figure(typ)
  plt.plot(train_statistics, label='train')
  plt.plot(test_statistics, label='test')
  plt.xlabel('Iterations')
  plt.ylabel(typ)
  plt.legend()
  plt.text(right/2, top-top/6, 'Final Train {} = {:.2f}'.format(typ, train_statistics[-1]), fontsize=10, color='blue')
  plt.text(right/2, top-top/4.5,  'Final Test {} = {:.2f}'.format(typ, test_statistics[-1]), fontsize=10, color='blue')
  plt.savefig('images/'+id+'_'+typ[:4]+'.png')


if __name__ == '__main__':
  plot_curve('Miss Classification rate', train_accuracy, test_accuracy)
  plot_curve('Loss Ssum', train_loss, test_loss)

