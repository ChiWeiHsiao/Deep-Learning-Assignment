from PIL import Image
import numpy as np

# read r,g,b arrays from file
id = ''
filename = 'statistics/'+id
filename += '.json'



r = np.zeros((512,512))
g = np.ones((512,512))
b = np.zeros((512,512)) - 0.5

rgbArray = np.zeros((512,512,3), 'uint8')
rgbArray[..., 0] = r*256
rgbArray[..., 1] = g*256
rgbArray[..., 2] = b*256
img = Image.fromarray(rgbArray)
img.save('myimg.jpeg')
