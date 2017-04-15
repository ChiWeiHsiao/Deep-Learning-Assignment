from PIL import Image
import numpy as np

# read r,g,b arrays from file
experiment_id = 'toy'
filename = 'statistics/'+experiment_id
filename += '.json'

with open(filename, 'r') as f:
  log = json.load(f)
  # Accuracy
  train_accuracy =  log['train_accuracy_per_epoch']
  test_accuracy = log['test_accuracy_per_epoch']
  # Image and Feature_maps
  original_image = log['original_image']
  fmap0_conv1 = log['feature_map_0']['conv1']
  fmap0_pool1 = log['feature_map_0']['pool1']
  fmap0_lrn1 = log['feature_map_0']['lrn1']
  fmap0_conv2 = log['feature_map_0']['conv2']
  fmap0_pool2 = log['feature_map_0']['pool2']
  fmap0_lrn2 = log['feature_map_0']['lrn2']
  fmap1_conv1 = log['feature_map_0']['conv1']
  fmap1_pool1 = log['feature_map_0']['pool1']
  fmap1_lrn1 = log['feature_map_0']['lrn1']
  fmap1_conv2 = log['feature_map_0']['conv2']
  fmap1_pool2 = log['feature_map_0']['pool2']
  fmap1_lrn2 = log['feature_map_0']['lrn2']
  # Parameters of model and training
  conv_filter = log['conv_filter']
  conv_stride = log['conv_stride']
  pool_kernel = log['pool_kernel']
  dropout = log['dropout']
  epochs = log['epoch']
  batch_size = log['batch_size']


def draw_fmap(fmap, filename):
  filename = experiment_id + '__' +filename + '.jpeg'
  #[-1, 16, 16, 64]
  r = fmap[]

  img.save(filename)

r = np.zeros((512,512))
g = np.ones((512,512))
b = np.zeros((512,512)) - 0.5

rgbArray = np.zeros((512,512,3), 'uint8')
rgbArray[..., 0] = r*256
rgbArray[..., 1] = g*256
rgbArray[..., 2] = b*256
img = Image.fromarray(rgbArray)
img.save(filename)


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

