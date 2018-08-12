import numpy as np
import pandas as pd
import keras 
import matplotlib.pyplot as plt

dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)


#creating a data structure with 60 timesteps and 1 output
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train,y_train=np.array(X_train),np.array(y_train)

#reshaping into 3d
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 

regressor=Sequential()
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))   #for last or final LSTM layer we need not give return_sequence=True as it will be False
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

#compilein the RNN'
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the RNN to the Training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)


#Test set
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values

dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)  
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)

inputs=sc.transform(inputs)

X_test=[]
for i in range(60,80):                     
    X_test.append(inputs[i-60:i,0])
    
X_test=np.array(X_test)
    
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_prices=sc.inverse_transform(predicted_stock_price)    #inverse transform to cnvert the values from minmax (0,1) to actual values

plt.plot(real_stock_price,color='Red',label='Real Google Stock Price')
plt.plot(predicted_stock_prices,color='Blue',label='Predicted Google Stock Prices')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



















