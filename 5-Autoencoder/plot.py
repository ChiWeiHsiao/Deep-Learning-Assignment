from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np

experiment_id = 'try'
filename = experiment_id
jsonfile = 'statistics/'+filename+'.json'

with open(jsonfile, 'r') as f:
  log = json.load(f)
  train_loss = log['train_loss']
  test_loss = log['test_loss']
  original_image = log['original_image']
  reconstruct_image = log['reconstruct_image']


def draw_img(img, layer_name):
  filename = str(experiment_id) + '_' + layer_name + '.jpeg'
  # Transform to np array
  img = np.array(img)
  if img.shape[0] == 1:
    img = img[0]
  img_size = img.shape[1]
  r, g, b = np.dsplit(img, 3)
  # Reshape: (img_size,img_size,1) --> (img_size,img_size)
  r = r[:,:,0]
  g = g[:,:,0]
  b = b[:,:,0]
  # normaolize value to [0,1]
  max_rgb = np.max(np.abs(r.flatten()), axis=0) + 1
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
def plot_loss(figure, title):
  # Find out border
  left = 0
  right = len(train_loss)
  top = np.amax( train_loss )
  bottom = 0

  plt.figure(figure)
  plt.title(title)
  plt.plot(train_loss, label='train')
  plt.plot(test_loss, label='test')
  plt.xlabel('Number of iterations')
  plt.ylabel('Least Square Loss')
  plt.legend()
  plt.text(right/2, top-top/6, 'Final Train loss = {:.2f}'.format(100*train_loss[-1]), fontsize=10, color='green')
  plt.text(right/2, top-top/4.5,  'Final Test loss  = {:.2f}'.format(100*test_loss[-1]), fontsize=10, color='green')
  plt.savefig(figure+'.png')


if __name__ == '__main__':
  loss_filename = str(experiment_id)+'_loss'
  plot_loss(loss_filename, 'learning curve')
  for i in range(len(original_image)):
    draw_img(original_image[i], 'original_image_{}'.format(i))
    draw_img(reconstruct_image[i], 'reconstruct_image_{}'.format(i))
  
