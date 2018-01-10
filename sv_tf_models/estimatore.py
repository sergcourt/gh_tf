# coding=utf-8
import tensorflow as tf
import numpy as np


# feature_column = tf.feature_column.numeric_column('x', shape=[1])

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

estimatore=tf.estimator.LinearRegressor(feature_columns=feature_columns)


#set tensors

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])


input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn( {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn( {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

estimatore.train(input_fn=input_fn, steps=500)

train_metrics = estimatore.evaluate(input_fn = train_input_fn)

eval_metrics = estimatore.evaluate(input_fn = eval_input_fn)


print('tranin_metrics: %r', train_metrics)
print('eval_metrics: %r', eval_metrics)


