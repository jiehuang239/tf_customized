#!/usr/bin/env python3
"""An example of how to use tf.Dataset in Keras Model"""
import tensorflow as tf   # only work from tensorflow==1.9.0-rc1 and after
import cv2 
import numpy as np
_EPOCHS      = 5
_NUM_CLASSES = 10
_BATCH_SIZE  = 128

def training_pipeline():
  # #############
  # Load Dataset
  # #############
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  #print(x_train[0])
  #print(type(x_train))
  #print(len(x_train))
 # cv2.imshow("first graph",x_train[0])
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
  training_set = tfdata_generator(x_train, y_train, is_training=True, batch_size=_BATCH_SIZE)
  testing_set  = tfdata_generator(x_test, y_test, is_training=False, batch_size=_BATCH_SIZE)

  # #############
  # Train Model
  # #############
  
  model = keras_model()  # your keras model here
  model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
  model.fit(
      training_set.make_one_shot_iterator(),
      steps_per_epoch=len(x_train) // _BATCH_SIZE,
      epochs=_EPOCHS,
      validation_data=testing_set.make_one_shot_iterator(),
      validation_steps=len(x_test) // _BATCH_SIZE,
      verbose = 1)
  #save model
  model_json = model.to_json()
  with open("model.json","w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")
  print("Save model to disk")



def tfdata_generator(images, labels, is_training, batch_size=128):
    '''Construct a data generator using tf.Dataset'''

    def preprocess_fn(image, label):
        '''A transformation function to preprocess raw data
        into trainable input. '''
        x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
        y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_training:
        dataset = dataset.shuffle(1000)  # depends on sample size

    # Transform and batch data at the same time
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        preprocess_fn, batch_size,
        num_parallel_batches=4,  # cpu cores
        drop_remainder=True if is_training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset



def keras_model():
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input

    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3),activation='relu', padding='valid')(inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(_NUM_CLASSES, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_test.shape)
    json_file = open("model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    print("loaded model from disk")
    loaded_model.compile("adam","categorical_crossentropy",metrics=["acc"]) 
    input_x = tf.placeholder(tf.float32,shape=[1,1,28,28,1])
    dataset = tf.data.Dataset.from_tensor_slices(input_x)
    iter = dataset.make_initializable_iterator()
    el = iter.get_next()
    with tf.Session() as sess:
      for i in range(10):
        print("***********")
        data =np.reshape( x_test[i],(1,1,28,28,1))
        sess.run(iter.initializer,feed_dict={input_x:data})
        cate = loaded_model.predict(sess.run(el))
        print(y_test[i])
        print(np.where(cate==np.amax(cate))[1])



