import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *

# test

train_data_df = pd.read_csv('~/Desktop/keras/Exercise Files/03/sales_data_training.csv')

test_data_df = pd.read_csv('~/Desktop/keras/Exercise Files/03/sales_data_test.csv')

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_train = scaler.fit_transform(train_data_df)

scaled_test = scaler.transform(test_data_df)

print(
    'total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8])')

scaled_train_data_f = pd.DataFrame(scaled_train, columns=train_data_df.columns.values)
scaled_test_data_f = pd.DataFrame(scaled_test, columns=test_data_df.columns.values)

scaled_train_data_f.to_csv('scaled_train_data.csv', index=False)

scaled_test_data_f.to_csv('scaled_test_data.csv', index=False)

trainer_data = pd.read_csv('scaled_test_data.csv')
tester_data = pd.read_csv('scaled_train_data.csv')

X = trainer_data.drop('total_earnings', axis=1).values

Y = trainer_data[['total_earnings']].values

model= Sequential ()


model.add(Dense(50, input_dim=9, activation= 'relu' ))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation= 'linear') )

model.compile(loss='mse', optimizer='adam')


model.fit(X, Y, epochs=50, shuffle=True, verbose=2)

X_test = tester_data.drop('total_earnings', axis=1).values

Y_test = tester_data[['total_earnings']].values

test_error_rate= model.evaluate(X_test,Y_test, verbose=0 )

print(' this is your test error rate {} '.format(test_error_rate))


X_predict= pd.read_csv("proposed_new_product.csv").values

prediction=model.predict(X_predict)

prediction = prediction [0] [0]

prediction = prediction +0.1159

prediction = prediction / 0.000036968

print('questo: ${}'.format(prediction))


model.save("k_diff_model_saved.h5")
print("model saved to disk")