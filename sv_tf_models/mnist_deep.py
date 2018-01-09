# coding=utf-8
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

K = 200
L = 100
M = 60
N = 30

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define shape of the sizes of inputs


X = tf.placeholder(dtype=tf.float32, shape= [None, 784 ])  #None for # of sample images, 28x28 img size, 1 for B/W or 3 for RGB


# define shape of the layers of the NN


W1 = tf.Variable(tf.truncated_normal([784, K], stddev=0.1)) # (784 by 10 matrix) 784 pix (28x28), 10 categories (#0to #9) to predict

b1 = tf.Variable(tf.zeros([K])) # a (1*10) vector to add for each category of the prediction in the W matrix

W2 = tf.Variable(tf.truncated_normal([K, L ], stddev=0.1))

b2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M ], stddev=0.1))

b3 =tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N ], stddev=0.1))

b4 =tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, 10 ], stddev=0.1))

b5 =tf.Variable(tf.zeros([10]))



sess.run( tf.global_variables_initializer())

# reshape and flatten X

X = tf.reshape(X, [-1, 784])

# create a prediction structuring a feed forward computation model layer by layer

Y1 = tf.nn.relu(tf.matmul(X, W1)+b1)

Y2 = tf.nn.relu(tf.matmul(Y1, W2)+b2)

Y3 = tf.nn.relu(tf.matmul(Y2, W3)+b3)

Y4 = tf.nn.relu(tf.matmul(Y3, W4)+b4)

Y= tf.nn.softmax(tf.matmul(Y4, W5)+b5)


# create placeholder for a set of labels

Y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10]) # none for # of sample label images, 10 for categories of right or wrong


# create a Loss function to do automatic reduction of the loss (hardcoding XENT)

xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))


#keeping track of % of correct prediction

is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1) )
accuracy = tf.reduce_mean(( tf.cast(is_correct, tf.float32)))


# optimizing

optimizer = tf.train.GradientDescentOptimizer(0.5)
train_step = optimizer.minimize(xent)


# kick off a session

sess = tf.Session()

# sess.run(init)


for i in range (10000):

    if  i % 5 ==0:
        batch = mnist.train.next_batch(100)

        train_step.run(feed_dict={X: batch[0], Y_: batch[1]})

        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))



sess.close()