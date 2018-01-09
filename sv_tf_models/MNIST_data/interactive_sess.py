# coding=utf-8
import tensorflow as tf


sess = tf.InteractiveSession()

a = tf.constant(4)
b = tf.constant(5)

c = a + b

print("c value is:", c.eval())
sess.close()
