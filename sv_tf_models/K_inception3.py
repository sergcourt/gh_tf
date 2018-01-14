import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3


model = inception_v3.InceptionV3()


img = image.load_img('cotco.jpg', target_size=(299,299) )


x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = inception_v3.preprocess_input(x)

prediction =model.predict(x)


predicted_classes= inception_v3.decode_predictions(prediction, top=10)

print("this image is: ")

for id, name, likelihood in predicted_classes [0]:
    print(" - {} : {:2f} likelihood".format(name,likelihood))