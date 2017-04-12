'''
Read statistics from tensorflow .event file
then output it as json file
'''
import tensorflow as tf
import json

# Read statistics from tensorflow .event file
train_accuracy = []
test_accuracy = []
step = 1
steps_per_epoch = 391
for e in tf.train.summary_iterator('../log/cnn_3/events.out.tfevents.1491932310.Aspire'):
#for e in tf.train.summary_iterator('../log/cnn_2/events.out.tfevents.1491821101.Aspire'):
#for e in tf.train.summary_iterator('../log/cnn_1/events.out.tfevents.1491744412.Aspire'):
  for v in e.summary.value:
    if v.tag == 'Accuracy':
      step +=  1
      if step % steps_per_epoch == 0:
        train_accuracy.append(v.simple_value)
    elif v.tag == 'Accuracy/Validation':
      test_accuracy.append(v.simple_value)

print('train_accuracy:', len(train_accuracy))
print('test_accuracy:', len(test_accuracy))




# Output it as json file
filename = 'accuracy_cnn_3'
filename += '.json'
with open(filename, 'w') as f:
  data = {
    'train_accuracy': train_accuracy,
    'test_accuracy' : test_accuracy 
  }
  json.dump(data, f)
