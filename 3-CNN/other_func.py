
def calculate_accuracy(sess, mode ='test'):
	acc = 0.0
	if mode == 'train':
		dataset = train_data
	elif mode == 'test':
		dataset = test_data
	# with tf.Session() as sess:
	for i in range(dataset.total_batch):
		next_x, next_y = dataset.next_batch()
		acc += sess.run(accuracy, feed_dict={x: next_x, y: next_y}).tolist()
		print('batch ',i, ' acc = ', acc)
	acc /= dataset.total_batch
	print('avg acc = ',acc)
	return acc
	
def extract_featuremap(sess, layer):
	channels = sess.run( layer, feed_dict={x: first_image, y:first_label})[0] 
	# channels contain multi channels of fmap
	# choose one from channels
	which = 0 
	fmap = channels[:,:,which]
	# preserve only four digit after decimal point
	fmap = np.around(fmap, decimals=4)
	tolist = fmap.tolist()
	return tolist

def extract_multiple_featuremaps(sess, name='feature_map_0'):
	log[name]['conv1'] = extract_featuremap(sess, conv1)
	log[name]['pool1'] = extract_featuremap(sess, pool1)
	log[name]['lrn1'] = extract_featuremap(sess, lrn1)
	log[name]['conv2'] = extract_featuremap(sess, conv2)
	log[name]['pool2'] = extract_featuremap(sess, pool2)
	log[name]['lrn2'] = extract_featuremap(sess, lrn2)
	return
