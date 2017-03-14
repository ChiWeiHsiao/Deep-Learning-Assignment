import numpy as np

LAYERS = 2
UNITS = 10 

mu, sigma = 0, 0.01

x = [[77,77],[0,0]]
weight = [ np.zeros_like(x[0], dtype=np.float32)]
print(weight)

for i in range(LAYERS):
    weight.append(np.zeros((UNITS), dtype=np.float32))

print('weight:',weight)
print('weight[1][0]+[1]:',weight[1][0]+weight[1][1])
