import matplotlib.pyplot as plt
import json
import numpy as np

id = 'L2_1'
print(id)
n_layers = 3

filename = 'statistics/'+id
filename += '.json'

with open(filename, 'r') as f:
  log = json.load(f)
  train_accuracy =  log['train_accuracy_per_epoch']
  test_accuracy = log['test_accuracy_per_epoch']

  weights = []
  weights.append(log['weight_1'])
  weights.append(log['weight_2'])
  if(n_layers >= 3):
    weights.append(log['weight_3'])
  if(n_layers >= 5):
    weights.append(log['weight_4'])
    weights.append(log['weight_5'])


# Show Learning Curve
def plot_accuracy(figure, title):
  # find out border
  left = 0
  right = len(train_accuracy)
  top = np.amax( np.array(train_accuracy) )
  bottom = 0

  plt.figure(figure)
  plt.title(title)
  plt.plot(train_accuracy, label='train')
  plt.plot(test_accuracy, label='test')

  plt.xlabel('Number of epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.text(right/2, 0.1075, 'Final Train Accuracy = {:.2f}%'.format(100*train_accuracy[-1]), fontsize=10, color='green')
  plt.text(right/2, 0.1100,  'Final Test Accuracy  = {:.2f}%'.format(100*test_accuracy[-1]), fontsize=10, color='green')
  plt.savefig(figure+'.png')

# Show weights histogram
def plot_hist_all(figure, title, weights):
  plt.figure(figure) #experiment_id
  plt.title(title)
  color = ['red', 'yellow', 'cyan', '#f20ca9', 'black']
  alpha = [0.6, 0.4, 0.4, 0.3, 0.15]
  for i in range(len(weights)):
    plt.hist(weights[i], bins=100, facecolor=color[i], alpha=alpha[i], label='h'+str(i+1), normed=1)
  plt.xlabel('value')
  plt.ylabel('count')
  plt.legend()
  #plt.show()
  plt.savefig(figure+'.png')

def plot_hist(figure, title, weight):
  plt.figure(figure) #experiment_id
  plt.title(title)
  plt.hist(weight, bins=60, facecolor='blue', alpha=1, label='h1')#, normed=1)
  plt.xlabel('value')
  plt.ylabel('count')
  plt.savefig(figure+'.png')

def plot_small_hist(figure, title, weights):
  fig, ax = plt.subplots(ncols=1)
  #xticks = np.arange(-1.0, 1.0, 0.001)
  ax.set_title(title)
  #ax.set_xticks(xticks)
  ax.grid(True)
  ax.set_xlim(-1e-7, 1e-7)
  color = ['red', 'yellow', 'cyan', '#f20ca9', 'black']
  alpha = [0.6, 0.4, 0.4, 0.3, 0.15]
  for i in range(len(weights)):
    ax.hist(weights[i], bins=1000, facecolor=color[i], alpha=alpha[i], label='h'+str(i+1), normed=1)
  plt.xlabel('value')
  plt.ylabel('count')
  ax.legend()
  #ax.show()
  plt.savefig(figure+'.png')

def count_exact_zero(weights):
  cnt = 0
  total = 0
  for w in weights:
    for i in w:
      total += 1
      if np.absolute(i) <= 1e-5:
        cnt += 1
      #if np.absolute(i) == 0:
        #cnt += 1
  print('Exactly zero: ', cnt)
  print('Total: ', total)

if __name__ == '__main__':
  #plot_accuracy('acc_'+id, 'learning curve')
  #plot_hist_all('hist_all_'+id, 'hisltogram_all', weights)
  count_exact_zero(weights)
  #for i in range(len(weights)):
    #plot_hist('hist_h'+str(i+1)+'_'+id, 'hisltogram_h'+str(i+1), weights[i])
  #plot_hist('hist_h1_'+id, 'hisltogram_h1', weight_1)
