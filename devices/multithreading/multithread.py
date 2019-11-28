#!/usr/bin/env python3
"""An example of how to use tf.Dataset in Keras Model"""
import time
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
import threading
import time
def thread_function(j,x_test):
    i=0
    json_file = open("model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    print("loaded model from disk")
    model.compile("adam","categorical_crossentropy",metrics=["acc"])
    print("compilation finished!")
    time.sleep(j*0.0007)
    start = time.time()
    while True:
      index = 10*i+j
     # index = i
      i+=1
      if(index>=100):
          end = time.time()
          print("thread {}".format(j))
          print("processing time = {}".format(end-start))
          break
      print("***********")
      cate = model.predict(np.reshape(x_test[index],(1,28,28,1)))
      print(y_test[index])
      print(np.where(cate==np.amax(cate))[1])

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    i = 0
    print(x_test.shape)
    #loaded_model.compile("adam","categorical_crossentropy",metrics=["acc"])
    lock=threading.Lock()
    list_of_thread = []
    for i in range(10):
      x=threading.Thread(target=thread_function,args=(i,x_test))
      x.start()
      list_of_thread.append(x)


