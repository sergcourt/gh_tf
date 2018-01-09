# coding=utf-8
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define shape of the sizes of all inputs

X = tf.placeholder(dtype=tf.float32, shape= [None, 784 ])  #None for # of sample images, 28x28 img size, 1 for B/W or 3 for RGB

W = tf.Variable(tf.zeros([784, 10])) # (784 by 10 matrix) 784 pix (28x28), 10 categories (#0to #9) to predict

b = tf.Variable(tf.zeros([10])) # a (1*10) vector to add for each category of the prediction in the W matrix

sess.run( tf.global_variables_initializer())


# create a prediction structuring a feed forward computation model

Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W)+b)



# create placeholder for a set of labels

Y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10]) # none for # of sample label images, 10 for categories of right or wrong


# create a Loss function to do automatic reduction of the loss (hardcoding XENT)

xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))


#keeping track of % of correct prediction

is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1) )
accuracy = tf.reduce_mean(( tf.cast(is_correct, tf.float32)))


# optimizing

optimizer = tf.train.GradientDescentOptimizer(0.03)
train_step = optimizer.minimize(xent)


# kick off a session

sess = tf.Session()

# sess.run(init)


for i in range (100):



    batch = mnist.train.next_batch(100)

    train_step.run(feed_dict={X: batch[0], Y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))



sess.close()

    """ 
    #load batch of imgs and  labels
    batch_X, batch_Y = mnist.train.next_batch (100)
    train_data = {X:batch_X, Y_:batch_Y}



    # train
    sess.run(train_step, feed_dict=train_data)






    #measure success and failure

    a,c = sess.run([train_step, xent], feed_dict=train_data)

    #meausre success on testing data

    test_data = {X:mnist.test.images, Y:mnist.test.labels}
    a,c = sess.run([accuracy, xent], feed_dict=test_data )
    """







