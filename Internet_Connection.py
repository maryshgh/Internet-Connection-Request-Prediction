# Predicting the internet connection request attempt in mobile networks using dense 

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from glob import glob
import tensorflow as tf

dataset_internet_interpolated = np.load('ID10002.npy')

#Normalizing data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_normalize = scaler.fit_transform(dataset_internet_interpolated)

# Part 2 - Building the ANN with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#create input sequence
train_window = 120
def create_inout_sequences(input_data, tw):
    input_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        input_seq.append(train_seq)
    return input_seq

train_input_seq = create_inout_sequences(train_data_normalize, train_window)
train_input= np.array(train_input_seq)
train_input = np.reshape(train_input,(train_input.shape[0],train_input.shape[1]))

train_output = train_data_normalize[train_window:,:]

# define base model

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(200, input_dim=train_window, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = 'accuracy')
	return model

# training network with just one time
estimator = baseline_model()
estimator.fit(train_input, train_output, batch_size = 30, epochs = 100)
#results = cross_val_score(estimator, train_input, test_input)
test_data_normalize=train_data_normalize[-2*train_window:,:]
test_input_seq = create_inout_sequences(test_data_normalize, train_window)
test_input= np.array(test_input_seq)
test_input = np.reshape(test_input,(test_input.shape[0],test_input.shape[1]))
test_output = test_data_normalize[-train_window:,:]
predicted_Internet = estimator.predict(test_input)

# evaluate model 
from sklearn.metrics import mean_squared_error
mean_squared_error(test_output, predicted_Internet)

from sklearn.metrics import r2_score
r2_score(test_output, predicted_Internet)


# Visualising the results
plt.plot(test_output, color = 'red', label = 'Real Internet connection requests')
plt.plot(predicted_Internet, color = 'blue', label = 'Predicted Internet connection requests')
plt.title('Internet connection requests Prediction')
plt.xlabel('Time')
plt.ylabel('Internet connection requests')
plt.legend()
plt.show()