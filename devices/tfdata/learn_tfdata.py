#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
x=tf.placeholder(tf.float32,shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices(x)#create dataset
iter = dataset.make_initializable_iterator()#create iterator
el = iter.get_next() #consume iterator
with tf.Session() as sess:
  for i in range(10):
      print("epoch {}".format(i))
      data = np.random.sample((1,2))
      print(data)
      sess.run(iter.initializer,feed_dict={x:data})
      print(sess.run(el))
    
