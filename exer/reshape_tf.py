
u = tf.constant([1, 2, 3], name='u')
print 'u:', u
v = tf.expand_dims(u, axis=0)
w = tf.expand_dims(u, axis=-1)
if v.shape == (1, 3):
  print 'v has correct shape (1, 3).'
if w.shape == (3, 1):
  print 'w has correct shape (3, 1).'

x = tf.constant([1, 2, 3, 4, 5, 6], name='x')
print '\nx:', x
y = tf.reshape(x, [2, 3, 1])
if y.shape == (2, 3, 1):
  print 'y has correct shape (2, 3, 1).'

a = tf.constant([[[1, 2]]], name='a')
b = tf.constant([[[3, 4]]], name='b')
print '\na:', a
print 'b:', b
c = tf.squeeze(tf.concat([a, b], axis=2))
if c.shape == (4,):
  print 'c has correct shape (4,).'
