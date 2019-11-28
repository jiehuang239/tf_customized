import tensorflow as tf
tf.load_library('/usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so.1')
rgb2grey_module = tf.load_op_library('./rgb2greyscale.so')
grey = rgb2grey_module.rgb2greyscale([21,234,45,234,12,67])
with tf.Session() as sess:
    sess.run(print(grey))
