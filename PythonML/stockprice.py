import pandas as pd
import quandl, math, datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as nm
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forcast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)


forecast_out = int(math.ceil(0.03*len(df)))
print(forecast_out)
df['label'] = df[forcast_col].shift(-forecast_out)
# df.dropna(inplace=True)

# print(df.head())

X = nm.array(df.drop(['label'], 1))

# [optional] scale features
X = preprocessing.scale(X)

X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = nm.array(df['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# # -1 to chho processor -1 will use minimum processors
# clf = LinearRegression(n_jobs=-1)
#
# # to fit our training set
# clf.fit(X_train, y_train)

# # storing classifier into pickle for infuture use without training data
# with open('stockpriceclassifier.pickel', 'wb') as f:
#     pickle.dump(clf, f)

# reading classifier from pickle
pckel_in = open('stockpriceclassifier.pickel', 'rb')
clf = pickle.load(pckel_in)

# to see out model's accuracy on testing data
accuracy = clf.score(X_test, y_test)

'''
here we will get out predicted value
'''
# forecast_set = clf.predict(X)
# print(forecast_set, accuracy, forecast_out)

forecast_set = clf.predict(X_lately)
# print(forecast_set, accuracy, fore    cast_out)


df['Forecast'] = nm.nan


# need to find out last date
'''
last_date will be given by df
then we will convert it into unix format
one_day = 86400 seconds
to find next unix we will add one day into last unix
'''
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


'''
populating date for each data
'''

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [nm.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()