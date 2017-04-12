import matplotlib.pyplot as plt
import json
'''
log = { 
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
filename = 'L2_1'
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

plt.figure('dnn_L2')
plt.title('learning curve')
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.show()


plt.figure('dnn_L2')
plt.title('histogram')
plt.plot(weight)
plt.xlabel('value')
plt.ylabel('count')
plt.show()

