from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
import numpy as np
import os

experiment_id = 'reconstruct-e500-onegaussian'
statitics_file = 'statistics/'+experiment_id+'.npz'
save_directory = 'results/'+experiment_id+'-1'

statistics = np.load(statitics_file)
reconstruction_loss = statistics['reconstruction_loss']
generator_loss = statistics['generator_loss'][1:]
discriminator_loss = statistics['discriminator_loss'][1:]
original_images = statistics['original_images']
encoded_images = statistics['encoded_images']
reconstruct_images = statistics['reconstruct_images']
encoded_feature_vector = statistics['encoded_feature_vector']
label = statistics['label']

'''
jsonfile = 'statistics/'+filename+'.json'
with open(jsonfile, 'r') as f:
    log = json.load(f)
    train_loss = log['train_loss']
    original_image = log['original_image']
    reconstruct_image = log['reconstruct_image']
'''
def draw_img(img, name):
    filename = save_directory + '/' + name + '.jpeg'
    plt.clf()
    img_size = int(np.sqrt(img.shape[0]))
    img = np.reshape(img, [img_size, img_size])
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.savefig(filename)


# Show Learning Curve
def plot_loss(loss, filename):
    filename = save_directory + '/' + filename
    # Find out border
    left = 0
    right = len(loss)
    top = np.amax( loss )
    bottom = 0
    # Build x axis
    x_axis = [i for i in range(0, len(loss))] 

    plt.clf()
    plt.title(filename)
    plt.plot(x_axis, loss)
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.text(right/9, top-top/6, 'Final Train loss = {:.10f}'.format(loss[-1]), fontsize=10, color='green')
    plt.savefig(filename+'.png')


def draw_tsne(z, label):
    tsne = TSNE(n_components = 2, random_state = 0)
    t_z = tsne.fit_transform(z)
    colors = cm.rainbow(np.linspace(0, 1, 10))
    scatter = []
    n_points = label.shape[0]
    n_class = 10
    index = range(n_points)
    plt.clf()
    #plot the t_z, color is determined by label
    for i in range(n_class):
        tmp = np.extract(np.equal(label, np.full((n_points, ), i)), index)
        scatter.append(plt.scatter(t_z[tmp, 0], t_z[tmp, 1], c = colors[i] ,s = 5))
    plt.legend(scatter, index)
    filename = save_directory + '/' + 'tsne.py'
    plt.savefig(filename+'.png')
    

if __name__ == '__main__':
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.figure(experiment_id)

    draw_tsne(encoded_feature_vector, label)

    plot_loss(reconstruction_loss, 'reconstruction_loss')
    plot_loss(generator_loss, 'generator_loss')
    plot_loss(discriminator_loss, 'discriminator_loss')
    
    num_images = 10
    for i in range(num_images):
        draw_img(original_images[i], str(i)+'_original')
        draw_img(encoded_images[i], str(i)+'_encoded')
        draw_img(reconstruct_images[i], str(i)+'_reconstruct')


    reconstruct_images = np.load('statistics/random_reconstruct_img.npz')['random_reconstruct_img']
    for i in range(10):
        draw_img(reconstruct_images[i], '_reconstruct_from_random'+str(i))


