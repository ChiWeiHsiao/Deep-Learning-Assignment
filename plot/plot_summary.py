import matplotlib.pyplot as plt
import json

# Read in it as json file
filename = 'accuracy_cnn_3'
filename += '.json'
with open(filename, 'r') as f:
  data = json.load(f)
  train =  data['train_accuracy']
  test = data['test_accuracy']
  print(train)

plt.figure('cnn_3')
#plt.title('learning curve')
plt.plot(train)
plt.plot(test)
plt.xlabel('Number of epochs')
#plt.ylabel('Loss')
plt.ylabel('Accuracy')
plt.show()
