import tensorflow as tf
import cPickle
import glob

filenames = glob.glob('./cifar-10-batches-py/data_batch_*')

print len(filenames)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def loading_data(filenames):
	data  = unpickle(filenames[0])['data']
	labels = unpickle(filenames[0])['labels']
	'''for file in filenames[1:]:
		data  += unpickle(file)['data']
		labels += unpickle(file)['labels']'''
	print len(data), len(labels)
	return data, labels

def batch_data(filenames):
	X, y= loading_data(filenames)
	images = tf.constant(X, dtype=tf.float32) 
	labels = tf.constant(y, dtype=tf.int32)
	data = tf.contrib.data.Dataset.from_tensor_slices((images,labels))
	data = data.repeat(30)
	data = data.batch(10)
	iterator = data.make_one_shot_iterator()
	batch_images, batch_labels = iterator.get_next()
	return batch_images, batch_labels


print unpickle(filenames[0])
#with tf.Session() as sess:
#	print sess.run(batch_data(filenames))