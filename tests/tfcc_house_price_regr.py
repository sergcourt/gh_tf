

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from numpy import linspace
from sklearn import metrics

from tensorflow.contrib.learn import *

print(tf.VERSION)


def preprocess_features(california_housing_dataframe):
  """This function takes an input dataframe and returns a version of it that has
  various features selected and pre-processed.  The input dataframe contains
  data from the california_housing data set."""
  # Select fewer columns to allow training a bit faster.
  output_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "households",
     "median_income"]].copy()
  output_features["roomsPerPerson"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return output_features




def preprocess_targets(california_housing_dataframe):
  """This function selects and potentially transforms the output target from
  an input dataframe containing data from the california_housing data set.
  The object returned is a pandas DataFrame."""
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets





# Set the output display to have one digit for decimal places, for display
# readability only.
pd.options.display.float_format = "{:.1f}".format



rough_house=pd.read_csv('/Users/sergiovillani/Desktop/tensorflow/tf_files/tests/housing.csv')


raw_training_df = rough_house.reindex(np.random.permutation(
  rough_house.index))



#print(raw_training_df.head())


# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(raw_training_df.head(12000))
training_targets = preprocess_targets(raw_training_df.head(12000))


# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(raw_training_df.tail(5000))
validation_targets = preprocess_targets(raw_training_df.tail(5000))





# Sanity check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())



#manually create feature columns



longitude = tf.contrib.layers.real_valued_column("longitude")
latitude = tf.contrib.layers.real_valued_column("latitude")
housingMedianAge = tf.contrib.layers.real_valued_column("housing_median_age")
households = tf.contrib.layers.real_valued_column("households")
medianIncome = tf.contrib.layers.real_valued_column("median_income")
roomsPerPerson = tf.contrib.layers.real_valued_column("roomsPerPerson")

feature_columns=set([
  longitude,
  latitude,
  housingMedianAge,
  households,
  medianIncome,
  roomsPerPerson])

print ("feature_columns created!")


LEARNING_RATE = 1.0
STEPS = 500
BATCH_SIZE = 100


#def train_model(learning_rate, steps, batch_size, feature_columns):


training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_examples,
                                                            y=training_targets['median_house_value'],
                                                            num_epochs=None,
                                                            batch_size=32,
                                                            shuffle=False)

predict_training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_examples,
                                                                    y=training_targets["median_house_value"],
                                                                    num_epochs=1,
                                                                    shuffle=False)

predict_validation_input_fn = tf.estimator.inputs.pandas_input_fn(x=validation_examples,
                                                                      y=validation_targets["median_house_value"],
                                                                      num_epochs=1,
                                                                      shuffle=False)



    # build linear regressor

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns='preprocessed_features',
    optimizer=tf.train.AdamOptimizer(LEARNING_RATE),
    # config=mlcc_config.create_config(steps)
    )

    # Train the model.


linear_regressor.train(input_fn=training_input_fn,
                         steps=500..as_integer_ratio())

print("Model training finished.")





    #training_predictions = list(linear_regressor.predict_scores(
    #    input_fn=predict_training_input_fn))
    #calibration_data["predictions"] = pd.Series(training_predictions)
    #calibration_data["targets"] = training_targets.values
    #display.display(calibration_data.describe())





train_model (LEARNING_RATE, STEPS, BATCH_SIZE, feature_columns)

'''
# clean_house=rough_house.drop('ocean_proximity', axis=1).values

columns=['longitude,latitude,,total_rooms,total_bedrooms,population,households,median_income,median_house_value']
df_house=pd.DataFrame(clean_house)

#print(rough_house.describe())
#print(clean_house.describe())
print(df_house.head())



rough_house= rough_house.reindex(np.random.permutation(rough_house.index))
#print(shuffle_df_house.describe())
#print('dataset loaded and randomized\n')


rough_house= rough_house[["total_bedrooms", "population", "median_house_value"]]




rough_house['median_house_value']=(rough_house['median_house_value']/1000.0)

print(rough_house.describe())


#print(rough_house.head())


#not sure the use of this
my_feature_name='total_bedrooms'

#passing total bedroooms to a var called my_feature
my_feature=rough_house[[my_feature_name]]

targets=rough_house['median_house_value']
steps=100





training_input_fn=tf.estimator.LinearRegressor(feature_columns=[my_feature,targets]) #num_epochs=None, batch_size=1)

prediction_input_fn = tf.estimator.LinearRegressor(feature_columns=[my_feature,targets]) #num_epochs=1, shuffle=False)

#feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
#    training_input_fn)





estimator=LinearRegressor(
    feature_columns=feature_columns,
    optimizer=tf.GradientDescentOptimizer(learning_rate=0.00001),
    gradient_clip_norm=5.0,
)

print ("Training model...")
estimator.fit(
    input_fn=training_input_fn,
    steps=steps,
)

print ("Model training finished.")

'''