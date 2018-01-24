# coding=utf-8
from __future__ import print_function
import tensorflow as tf



#NODE= TENSOR OR  OPERATION that defines  NN and will be  encapsulated by a graph at executions


node1=tf.constant(7.0)
node2=tf.constant(8.0)

node3=tf.add(node1, node2)



print('node3: ', node3)




# shape NODE (or graph) to take external inputs
node4 = tf.placeholder(dtype=tf.float32)
node5 = tf.placeholder(dtype=tf.float32)

summer=tf.multiply(tf.multiply(node4,node5), node3)

#add variables to the equation: best way to represent shared persitent state manipulated by the model.
# Variables store  persistent tensor. Specific ops can read and modify it.
# you define them  (in capital V) with TYPE and initial VALUE such as this.
# Variables -unlike tf.tensors- exist outside the context of a single session.run call

W = tf.get_variable('my_first_var', None, dtype=tf.float32)




#set up and kick off session ans session has a capital S, like Variable has capital V

sess=tf.Session()
print('eccolo qui il nodo3:', sess.run(node3))
print(sess.run(summer,feed_dict={node4:[3,2], node5:[2,3]}))





#GRAPH= set of TF opeatioons represented by nodes.Session encapsulate CONTROL AD sTATE of TF runtime at execution.
'''
sess=tf.Session()
gino= sess.run([node1, node2])
print(gino)
'''