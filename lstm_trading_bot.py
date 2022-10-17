# -*- coding: utf-8 -*-
"""LSTM Trading Bot"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd

# %matplotlib inline
plt.style.use('fivethirtyeight')

from sklearn.metrics import mean_squared_error

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from datetime import datetime

symbol = 'SRMBNB'

!pip install python-binance
!pip install tpot

from binance.client import Client

import json
with open('creds.json') as f:
  data = json.load(f)

client = Client(data['key'],data['secret'])

candles = client.get_klines(symbol=symbol,interval=Client.KLINE_INTERVAL_1MINUTE)

len(candles)

candles[499]

price = np.array([float(candles[i][4]) for i in range(500)])

time = np.array([int(candles[i][0]) for i in range(500)])

t = np.array([datetime.fromtimestamp(time[i]/1000).strftime('%H:%M:%S') for i in range(500)])

price.shape

plt.figure(figsize=(8,5))
plt.xlabel('Time Step')
plt.ylabel('Bitcoin Price $')
plt.plot(price)

timeframe = pd.DataFrame({'Time':t,'Price $BTC':price})

timeframe #minute by minute price

price = price.reshape(500,1)

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

scaler.fit(price[:374])

price = scaler.transform(price)

df = pd.DataFrame(price.reshape(100,5), columns = ['First', 'Second', 'Third', 'Fourth', 'Target'])

df.head()

#75% train, 25% test 

x_train = df.iloc[:74, :4]
y_train = df.iloc[:74, -1]

x_test = df.iloc[75:99, :4]
y_test = df.iloc[75:99, -1]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_train.shape, x_test.shape

model = Sequential()

model.add(LSTM(20, return_sequences=True, input_shape=(4, 1)))
model.add(LSTM(40, return_sequences=False))
model.add(Dense(1, activation = 'linear'))
model.compile(loss='mse', optimizer = 'rmsprop')

model.summary()

model.fit(x_train, y_train, batch_size=5, epochs = 100)

y_pred = model.predict(x_test)

plt.figure(figsize=[8,5])
plt.title('Model Fit')
plt.xlabel('Time Step')
plt.ylabel('Normalized Price')
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Prediction')
plt.legend()

plt.figure(figsize=[8,5])
plt.title('model fit')
plt.xlabel('time step')
plt.ylabel('price')
plt.plot(scaler.inverse_transform(y_test), label='True')
plt.plot(scaler.inverse_transform(y_pred), label='Prediction')
plt.legend()

testScore = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred)))
print('Test score: %.2f RMSE' % (testScore))

from sklearn.metrics import r2_score

print('RSsquared :', '{:.2%}'.format(r2_score(y_test, y_pred)))

model.save("Bitcoin_model.h5")

#from keras.models import load_model

# load model
#model = load_model('Bitcoin_model.h5')

# summarize model
#model.summary()

"""# **Second Model**"""

from sklearn.svm import SVR

#75% train, 25% test 

trainX = df.iloc[:74, :4]
trainY = df.iloc[:74, -1]

testX = df.iloc[75:99, :4]
testY = df.iloc[75:99, -1]

svr_linear = SVR(kernel = 'linear', C=1e3, gamma = 0.1)
svr_linear.fit(trainX, trainY)

predY = svr_linear.predict(testX)

plt.figure(figsize=[8,5])
plt.title('Model Fit')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.plot(scaler.inverse_transform(testY), label='True')
plt.plot(scaler.inverse_transform(predY), label='Prediction')
plt.legend()

testScore = np.sqrt(mean_squared_error(scaler.inverse_transform(testY), scaler.inverse_transform(predY)))
print('Test score: %.2f RMSE' % (testScore))

print('RSsquared : ',  '{:.2%}'.format(r2_score(testY, predY)))

"""# **Hyperparameter Tuning**"""

param_grid = {"C": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4 ],
			  "gamma": np.logspace (-2, 2, 50),
			  'epsilon':[0.1,0.2,0.5,0.3]}

from sklearn.model_selection import RandomizedSearchCV

svm_model = SVR(kernel='linear')

grid_search = RandomizedSearchCV(svm_model, param_grid, scoring = 'r2', n_jobs=-1 )

grid_search.fit(trainX, trainY)

print(grid_search.best_estimator_)

svm_model = SVR(C=10000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=100.0, kernel = 'linear', max_iter = -1, shrinking=True, tol = 0.001, verbose=False)

svm_model.fit(trainX, trainY)

pred = svm_model.predict(testX)

testScore = np.sqrt(mean_squared_error(scaler.inverse_transform(testY), scaler.inverse_transform(pred)))
print('Test Score: %.2f RMSE' % (testScore))

print('RSsquared :', '{:.2%}'.format(r2_score(testY,pred)))

plt.figure(figsize=[8,5])
plt.title('model fit')
plt.xlabel('time step')
plt.ylabel('price')
plt.plot(scaler.inverse_transform(testY), label='True')
plt.plot(scaler.inverse_transform(pred), label='Prediction')
plt.legend()

"""# **Ridge Regression**"""

from sklearn.linear_model import RidgeCV

ridge = RidgeCV()

ridge.fit(trainX, trainY)

Rpred = ridge.predict(testX)

testScore = np.sqrt(mean_squared_error(scaler.inverse_transform(testY), scaler.inverse_transform(Rpred)))
print('Test Score: ', testScore)

print('RSquared: ', '{:.2%}'.format(r2_score(testY, Rpred)))

plt.figure(figsize=[8,5])
plt.title('model fit')
plt.xlabel('time step')
plt.ylabel('price')
plt.plot(scaler.inverse_transform(testY), label='True')
plt.plot(scaler.inverse_transform(Rpred), label='Prediction')
plt.legend()

"""# **Hyperparameter Tuning**"""

normal_price = np.array([float(candles[i][4]) for i in range(500)])

data = pd.DataFrame(normal_price.reshape(100,5), columns = ['first', 'second', 'third', 'fourth', 'target' ])

data.head()

data.tail()

#75% train, 25% test

x_train_r = df.iloc[:74, :4]
y_train_r = df.iloc[:74, -1]

x_test_r = df.iloc[75:99, :4]
y_test_r = df.iloc[75:99, -1]

from tpot import TPOTRegressor

tpot = TPOTRegressor(generations = 5, population_size =50, verbosity = 2)
tpot.fit(x_train_r, y_train_r)

tpred = tpot.predict(x_test_r)

testScore = np.sqrt(mean_squared_error(y_test_r, tpred))
print('Test Score: %.2f RMSE' % (testScore))

print('RSquared :','{:.2%}'.format(r2_score(y_test_r,tpred)))

tpot.export('bitcoin.py')

plt.figure(figsize=[8,5])
plt.title('Model Fit')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.plot(np.array(y_test_r).reshape(24,),label='True')
plt.plot(tpred,label='Prediction')
plt.legend()

"""# **Trading Bot**"""

check = client.get_klines(symbol=symbol,interval=Client.KLINE_INTERVAL_1MINUTE)

check[499]

index = [496,495,498,499]

candles = scaler.transform(np.array([float(check[i][4]) for i in index]).reshape(1,-1))

model_feed = candles.reshape(1,4,1)

scaler.inverse_transform(model.predict(model_feed)[0])[0]

# trading bot

quantity = '0.05' # quantity to trade

order = False

index = [496,495,498,499]

# Sample Stuff
account = 500
risk = 0.05
asset = 0
last_account = 0
start_account = account

while True:
  price = client.get_recent_trades(symbol=symbol)
  candle = client.get_klines(symbol=symbol,interval=client.KLINE_INTERVAL_1MINUTE)
  candles = scaler.transform(np.array([float(candle[i][4]) for i in index]).reshape(1,-1))
  model_feed = candles.reshape(1,4,1)
  
  if order == False and float(price[len(price)-1]['price']) < float(scaler.inverse_transform(model.predict(model_feed)[0])[0]):
    #client.order_market_buy(symbol=symbol,quantity=quantity)
    order = True
    buy_price = client.get_order_book(symbol=symbol)['asks'][0][0]
    print('Buy @Market Price :',float(buy_price),' Timestamp :',str(datetime.now()))

    last_account = account
    asset = (account*risk)/float(buy_price)

    # Binance Fee of 0.1%
    asset = asset-asset*0.001

    account -= account*risk
  elif order == True and float(price[len(price)-1]['price'])-float(buy_price) >= 10:
    #client.order_market_sell(symbol=symbol,quantity=quantity)
    order = False
    sell_price = client.get_order_book(symbol=symbol)['bids'][0][0]

    print('Sell @Market Price :',float(sell_price),' Timestamp :',str(datetime.now()))

    # Binance Fee of 0.1%
    sold_for = asset*float(sell_price)
    sold_for = sold_for-sold_for*0.001

    account += sold_for
    asset = 0

    # if account > last_account:
    #   print('Trade Change: +',(1-last_account/account)*100,'%')
    # else:
    #   print('Trade Change: -',(1-account/last_account)*100,'%')

    # print('Overall Change: $',account-start_account)
    # print('Account Size: ',account)

    last_account = 0
  else:
    pass
