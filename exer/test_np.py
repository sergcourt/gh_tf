
import numpy as np

v = np.array([1, 2, 3])
print 'v:', v


v = np.expand_dims(v,axis=0)
w = np.reshape(v, [1,3,1] )
if v.shape == (1, 3):
  print 'v has correct shape (1, 3).'
if w.shape == (3, 1):
  print 'w has correct shape (3, 1).'

if w.shape == (1, 3, 1):
  print 'w has correct shape (1, 3, 1).'

x = np.array([1, 2, 3, 4, 5, 6])
print '\nx:', x
y = np.reshape(x,[ 2, 3, 1])
if y.shape == (2, 3, 1):
  print 'y has correct shape (2, 3, 1).'


a = np.array([[[1, 2]]])
b = np.array([[[3, 4]]])
print '\na:', a
print 'b:', b
c = a + b  # Combine a and b into one array of shape [4] using np.concatenate and np.squeeze.
if c.shape == (4,):
  print 'c has correct shape (4,).'
