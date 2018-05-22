import tensorflow as tf
import numpy as np
import pandas as pd

print (tf.__version__)


x = np.random.sample((100,2))


lino= pd.read_csv('/usr/local/google/home/sergiovillani/PycharmProjects/ghrep/gh_tf/exer/03/sales_data_training.csv', usecols=[1,2,3,4,5,6,7])

lin= lino.values

labels=pd.read_csv('/usr/local/google/home/sergiovillani/PycharmProjects/ghrep/gh_tf/exer/03/sales_data_training.csv',usecols=[0]  )

labelline=labels.values

#linuzzo= tf.data.Dataset.from_tensor_slices(lin)


dataset=tf.data.Dataset.from_tensors((labelline,lin))

print(dataset.output_shapes)

