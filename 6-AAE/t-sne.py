import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.manifold import TSNE

#z = np.random.randn(50, 10, 2)
#z[:,:,1] = np.random.randint(11, size=(50))
data = np.load('statistics/try-encoded-data.npz')
z = data['feature_vector'] #(2000, 400)
label = data['label'] #(2000, )
print('label', label.shape)
'''
z is the output of encoder
z = model.encoding(sess, data)
'''
tsne = TSNE(n_components = 2, random_state = 0)
t_z = tsne.fit_transform(z)
print('t_z:', t_z.shape)

'''
plot the t_z, color is determined by label
'''
colors = cm.rainbow(np.linspace(0, 1, 10))
print('color:', colors.shape)
scatter = []
n_points = label.shape[0]
n_class = 10
index = range(n_points)
for i in range(n_class):
    #tmp = np.where(indexes == i)
    tmp = np.extract(np.equal(label, np.full((n_points, ), i)), index)
    scatter.append(plt.scatter(t_z[tmp, 0], t_z[tmp, 1], c = colors[i] ,s = 5))

plt.legend(scatter, index)
plt.show()
