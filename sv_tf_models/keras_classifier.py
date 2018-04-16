import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
# %matplotlib inline


from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv3D, MaxPooling3D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils


train_dir = '/usr/local/google/home/sergiovillani/PycharmProjects/tf_test/inputs/train/'
test_dir = '/usr/local/google/home/sergiovillani/PycharmProjects/tf_test/inputs/test/'


os.listdir(train_dir)


ROWS = 64
COLS = 64
CHANNELS = 3



train_images= [train_dir+i for i in os.listdir(train_dir)]

train_dogs= [train_dir+i for i in  os.listdir(train_dir) if 'dog' in i ]

train_cats= [train_dir+i for i in  os.listdir(train_dir) if 'cat' in i ]

test_images= [test_dir+i for i in  os.listdir(test_dir)]


train_images =train_dogs[:1000]+train_cats [:1000]

random.shuffle(test_images)


test_images=test_images [:25]


def read_image(file_path):

    img=cv2.imread(file_path, cv2.IMREAD_COLOR)

    return cv2.resize (img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)



def prep_data(images):

    count= len(images)
    data= np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):

        image = read_image(image_file)
        data[i] = image.T
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))

    return data

train = prep_data(train_images)
test = prep_data(test_images)


print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))


train_labels= []
test_labels=[]


for i in train_images:
    if 'dog' in i:
        train_labels.append(i)

    else:train_labels.append(0)


for i in test_images:
    if 'dog' in i:
        test_labels.append(i)

    else:test_labels.append(0)

nClasses=2

# visualiz3 the 2, now labeled, datasets
# sns.countplot(labels)
# sns.plt.tile('cats & dogs')

print('train_labels shape : {}'.format(train_labels.__len__()))


print('test_labels shape : {}'.format(test_labels.__len__()))


def createModel():
    model = Sequential()
    #model.add(Convolution3D( ))
    model.add(Conv3D(64, (3, 3,3),strides=(1, 1, 1),padding='same', activation='relu', input_shape=train.shape))
    model.add(Conv3D(64, (3, 3,3),activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2,2),strides=(1,2,2)))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, (3, 3,3), padding='same', activation='relu'))
    model.add(Conv3D(64, (3, 3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2,2),strides=(2,2,2)))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, (3, 3,3), padding='same', activation='relu'))
    model.add(Conv3D(64, (3, 3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2,2),strides=(2,2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model


model1 = createModel()
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model1.fit(train, labels, batch_size=batch_size, epochs=epochs, verbose=1,
                     )

model1.evaluate(test,)































