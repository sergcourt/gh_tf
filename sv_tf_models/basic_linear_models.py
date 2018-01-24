# coding=utf-8
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


import tensorflow as tf

display_steps=5


# set hyperparameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# define inputs

X = tf.placeholder(tf.float32)

#defaine labels

Y = tf.placeholder(tf.float32)

#fundamental model equations

linear_model = W * X + b

# create a loss function

squared_deltas = tf.squared_difference(linear_model,Y)
cost = tf.reduce_mean(squared_deltas)

# SHape optimizations method

optimizer = tf.train.GradientDescentOptimizer(0.01)

#apply optimization to the LOSS and tell it what action to perform (MINIMIZE)

train = optimizer.minimize(cost)


#set training data

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]



#create a variables intializer function called "init"

init = tf.global_variables_initializer()

#Create conditions for tensorboard

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()



with tf.Session () as session:

    #initiallize variables
    session.run(tf.global_variables_initializer())


    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.

    log_writer = tf.summary.FileWriter('./logs', session.graph)

    # SET for LOOP FOR NUMBER Of ITERATIONS ON WHICH TO OPERATE THE OPTIMIZER

    for i in range(1500):

        # tell it what to actually perform in each loop (run 'train' which calls all the other functions it depends on
        # and pass in  a dictionary for the placeholderS (input and label)

        session.run(train, feed_dict={X: x_train, Y: y_train})

        if i % display_steps == 0:

            # create objects to visualize the progress of the training
            current_cost , current_summary = session.run([cost, summary], feed_dict={X: x_train, Y: y_train})

            log_writer.add_summary(current_cost, current_summary)


            print('W: %s, b: %s, loss: %s' % (current_cost))












