from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np
import os

experiment_id = 'try'
directory = 'results/'+experiment_id
filename = experiment_id
jsonfile = 'statistics/'+filename+'.json'
with open(jsonfile, 'r') as f:
    log = json.load(f)
    train_loss = log['train_loss']
    original_image = log['original_image']
    reconstruct_image = log['reconstruct_image']

def draw_img(img, name):
    filename = directory + '/' + name + '.jpeg'
    # Resize
    img = np.array(img)
    img_size = int(np.sqrt(img.shape[0]))
    img = np.reshape(img, [img_size, img_size])
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    #plt.savefig('try.png')
    plt.show()


# Show Learning Curve
def plot_loss(filename, title):
    filename = directory + '/' + filename
    # Find out border
    left = 0
    right = len(train_loss)
    top = np.amax( train_loss )
    bottom = 0
    # Build x axis
    x_axis = [100*i for i in range(0, len(train_loss))] 

    plt.figure(filename)
    plt.title(title)
    plt.plot(x_axis, train_loss)
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.text(right/2, top-top/6, 'Final Train loss = {:.2f}'.format(train_loss[-1]), fontsize=10, color='green')
    plt.savefig(filename+'.png')


if __name__ == '__main__':
    if not os.path.exists(directory):
        os.makedirs(directory)
    #plot_loss('loss', 'learning curve')
    
    try_img = [0 for i in range(784)] 
    try_img[0:100] = [0.5 for i in range(100)]
    draw_img(try_img, 'try_img')
