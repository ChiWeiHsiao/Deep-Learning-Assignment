import matplotlib.pyplot as plt
import json
'''
log = { 
  'experiment_id': experiment_id,
  'train_accuracy_per_epoch': [], 
  'test_accuracy_per_epoch': [], 
  'weight_1': [], 
  'weight_2': [], 
  'weight_3': [], 
  'bias_1': [], 
  'bias_2': [], 
  'bias_3': []
}
'''
id = 'L2_4'

filename = 'statistics/'+id
filename += '.json'

with open(filename, 'r') as f:
  log = json.load(f)
  train_accuracy =  log['train_accuracy_per_epoch']
  test_accuracy = log['test_accuracy_per_epoch']
  weight_1 = log['weight_1']
  weight_2 = log['weight_2']
  weight_3 = log['weight_3']
  bias_1 = log['bias_1']
  print(bias_1)
  bias_2 = log['bias_2']
  bias_3 = log['bias_3']

# Show Learning Curve
def plot_accuracy(figure, title):
  plt.figure(figure)
  plt.title(title)
  plt.plot(train_accuracy, label='train')
  plt.plot(test_accuracy, label='test')
  plt.xlabel('Number of epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.text(27, .25, 'Final Train Accuracy = {:.2f}%'.format(100*train_accuracy[-1]), fontsize=10, color='green')
  plt.text(27, .3,  'Final Test Accuracy  = {:.2f}%'.format(100*test_accuracy[-1]), fontsize=10, color='green')
  plt.show()

# Show weights histogram
def plot_hist_all(figure, title, weights):
  plt.figure(figure) #experiment_id
  plt.title(title)
  plt.hist(weights[0], bins=100, facecolor='red', alpha=0.6, label='h1', normed=1)
  plt.hist(weights[1], bins=100, facecolor='yellow', alpha=0.4, label='h2', normed=1)
  plt.hist(weights[2], bins=100, facecolor='cyan', alpha=0.4, label='h3', normed=1)
  plt.xlabel('value')
  plt.ylabel('count')
  plt.legend()
  plt.show()

def plot_hist(figure, title, weight):
  plt.figure(figure) #experiment_id
  plt.title(title)
  plt.hist(weight, bins=60, facecolor='blue', alpha=1, label='h1')#, normed=1)
  plt.xlabel('value')
  plt.ylabel('count')
  plt.show()

if __name__ == '__main__':
  plot_accuracy('acc_'+id, 'learning curve')
  plot_hist_all('hist_all_'+id, 'hisltogram_all', [weight_1, weight_2, weight_3])
  plot_hist('hist_h1_'+id, 'hisltogram_h1', weight_1)
  plot_hist('hist_h2_'+id, 'hisltogram_h2', weight_2)
  plot_hist('hist_h3_'+id, 'hisltogram_h3', weight_3)

