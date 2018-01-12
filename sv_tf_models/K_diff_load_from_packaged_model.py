import pandas as pd
from keras.models import load_model

model= load_model('k_diff_model_saved.h5')

x= pd.read_csv('proposed_new_product.csv').values

prediction= model.predict (x)

prediction= [0][0]

prediction =prediction +0.1159

prediction =prediction / 0.000036968

print('new product est value ${} :'.format (prediction))
