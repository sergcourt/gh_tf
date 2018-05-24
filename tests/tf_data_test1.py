import tensorflow as tf
import numpy as np
import pandas as pd

print (tf.__version__)


x = np.random.sample((100,2))


lino= pd.read_csv('/usr/local/google/home/sergiovillani/PycharmProjects/ghrep/gh_tf/exer/03/sales_data_training.csv', usecols=[1,2,3,4,5,6,7])

lin= lino.values

labels=pd.read_csv('/usr/local/google/home/sergiovillani/PycharmProjects/ghrep/gh_tf/exer/03/sales_data_training.csv',  usecols=[0]  )

labelline=labels.values

#linuzzo= tf.data.Dataset.from_tensor_slices(lin)


dataset=tf.data.Dataset.from_tensor_slices((labelline,lin))

print(dataset.output_shapes)

from google.colab import files

x= files.upload()

#x=pd.read_csv( '/usr/local/google/home/sergiovillani/PycharmProjects/ghrep/gh_tf/exer/03/sales_data_training.csv', usecols=[1,2,3,4,5,6,7])




x.shay=pd.read_csv('sales_data_training.csv', usecols=[8])
y.shapepe



dataset=tf.data.Dataset.from_tensor_slices((x,y))

#make a 0NE SHOT ITERATOR

iter=dataset.make_one_shot_iterator()

#call the itearator

guappo=iter.get_next()




with tf.Session() as sess:
  print ( sess.run ( guappo))



#tf.reset_default_graph()


#INITIALIZIBLE ITERATOR

#CREATE  A PLACEHOLDER

x_1=tf.placeholder(dtype=tf.float32,shape=[None,7])
y_1= tf.placeholder(dtype=tf.float32,shape=[None,1])




initializable_dataset=tf.data.Dataset.from_tensor_slices((x_1,y_1))

initializable_iterator=initializable_dataset.make_initializable_iterator()

cippo=initializable_iterator.get_next()

with tf.Session() as sess:
  sess.run (initializable_iterator.initializer, feed_dict={x_1:x, y_1:y})
  print(sess.run(cippo))



EPOCHS = 10
x = pd.read_csv('sales_data_training.csv', usecols=[1, 2, 3, 4, 5, 6, 7])
y = pd.read_csv('sales_data_training.csv', usecols=[8])

x_2, y_2 = tf.placeholder(dtype=tf.float32, shape=[None, 7]), tf.placeholder(dtype=tf.float32, shape=[None, 1])

_dataset = tf.data.Dataset.from_tensor_slices((x_2, y_2))

iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer, feed_dict={x_2: x, y_2: y})

    for i in range(EPOCHS):
        sess.run([features, labels])

    # sess.run(iter.initializer, feed_dict={x_2:x[0], y_2:y[0]}  )

    print(sess.run([features, labels]))
