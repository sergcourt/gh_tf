import tensorflow as tf

# Writes 4 TFExamples as TFRecord into the file.
train_filename = './tfrecord.out'


# 4 Examples with 3 features.
feature1 = [[11], [13], [18], [20]]
feature2 = ['a', 'b', 'c' , 'd']
#feature2 = [[1, 2,], [3 , 4]]
feature3 = [[1, 2], [2, 3], [3, 4], [4, 5]]

with tf.python_io.TFRecordWriter('./tfrecord.out') as writer:
  for f1, f2, f3 in zip(feature1, feature2, feature3):
    feature = {
        'feature1': tf.train.Feature(int64_list=tf.train.Int64List(value=f1)),
        'feature2': tf.train.Feature(bytes_list=tf.train.BytesList.FromString(value=f2)),
        #'feature2': tf.train.Feature(int64_list=tf.train.Int64List(value=f2)),
        'feature3': tf.train.Feature(int64_list=tf.train.Int64List(value=f3))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())