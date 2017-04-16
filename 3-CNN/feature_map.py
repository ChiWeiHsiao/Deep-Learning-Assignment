from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np

# read r,g,b arrays from file
experiment_id = 'first'
  #filename = 'statistics/'+experiment_id
filename = experiment_id
jsonfile = filename+'.json'

with open(jsonfile, 'r') as f:
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
  conv1_filter = log['conv1_filter']
  conv2_filter = log['conv2_filter']
  conv_stride = log['conv_stride']
  pool_kernel = log['pool_kernel']
  dropout = log['dropout']
  epochs = log['epoch']
  batch_size = log['batch_size']


def draw_fmap(fmap, layer_name):
  filename = str(experiment_id) + '__' + layer_name + '.jpeg'
  # [1, img_size, img_size, 3]
  # [which_image=0, :, :, rgb=0/1/2 ]
  # Transform to np array
  fmap = np.array(fmap)
  print('fmap:', fmap.shape)
  if fmap.shape[0] == 1:
    fmap = fmap[0]
  img_size = fmap.shape[1]
  r, g, b = np.dsplit(fmap, 3)
  print('r:', r.shape)

  # Reshape: (img_size,img_size,1) --> (img_size,img_size)
  r = r[:,:,0]
  g = g[:,:,0]
  b = b[:,:,0]

  # normaolize value to [0,1]
  max_rgb = np.max(np.abs(r.flatten()), axis=0) + 1
  print('max rgb: ', max_rgb)
  r = r / max_rgb
  g = g / max_rgb
  b = b / max_rgb 
  
  # Create image
  rgbArray = np.zeros((img_size,img_size,3), 'uint8')
  rgbArray[..., 0] = r*256
  rgbArray[..., 1] = g*256
  rgbArray[..., 2] = b*256
  img = Image.fromarray(rgbArray)
  #img.show()
  img.save(filename)


# Show Learning Curve
def plot_accuracy(figure, title):
  # Find out border
  left = 0
  right = len(train_accuracy)
  top = np.amax( train_accuracy )
  bottom = 0

  plt.figure(figure)
  plt.title(title)
  plt.plot(train_accuracy, label='train')
  plt.plot(test_accuracy, label='test')
  plt.xlabel('Number of epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  # Annotate best acc
  best_index = np.argmax( np.array(train_accuracy) )
  best_train_acc = train_accuracy[best_index]
  best_test_acc = train_accuracy[best_index]
  plt.annotate('best train: %.2f\nbest test: %.2f' %(best_train_acc, best_test_acc), 
      xy=(best_index, best_train_acc), xytext=(best_index+right/10, top-top/10),
      arrowprops=dict(facecolor='black', shrink=0.05))

  plt.text(right/2, top-top/6, 'Final Train Accuracy = {:.2f}%'.format(100*train_accuracy[-1]), fontsize=10, color='green')
  plt.text(right/2, top-top/4.5,  'Final Test Accuracy  = {:.2f}%'.format(100*test_accuracy[-1]), fontsize=10, color='green')
  plt.savefig(figure+'.png')


if __name__ == '__main__':
  acc_filename = str(experiment_id)+'__acc'
  plot_accuracy(acc_filename, 'learning curve')
  draw_fmap(original_image, 'original_image')
  
  draw_fmap(fmap0_conv1[0], 'fmap0_conv1')
  draw_fmap(fmap0_pool1[0], 'fmap0_pool1')
  draw_fmap(fmap0_lrn1[0], 'fmap0_lrn1')
  draw_fmap(fmap0_conv2[0], 'fmap0_conv2')
  draw_fmap(fmap0_pool2[0], 'fmap0_pool2')
  draw_fmap(fmap0_lrn2[0], 'fmap0_lrn2')


  draw_fmap(fmap1_conv1[0], 'fmap1_conv1')
  draw_fmap(fmap1_pool1[0], 'fmap1_pool1')
  draw_fmap(fmap1_lrn1[0], 'fmap1_lrn1')
  draw_fmap(fmap1_conv2[0], 'fmap1_conv2')
  draw_fmap(fmap1_pool2[0], 'fmap1_pool2')
  draw_fmap(fmap1_lrn2[0], 'fmap1_lrn2')
  