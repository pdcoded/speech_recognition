import tensorflow as tf
a = tf.placeholder(tf.float32)
b= tf.placeholder(tf.float32)
def add(a,b):
	return a*b

x = add(a,b)
with tf.Session() as sess:
	print sess.run(x,feed_dict={a:5.,b:10.})