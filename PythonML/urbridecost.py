import pandas as pd
import quandl, math, datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as nm
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# df = quandl.get('WIKI/GOOGL')
df = pd.read_csv('ridecost.csv', index_col=0)

# print(df.head())

forcast_col = 'cost_without_service_tax'
df.fillna(-99999, inplace=True)


forecast_out = int(math.ceil(0.11*len(df)))
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

# print(len(X), len(y))  total rides data


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# -1 to chho processor -1 will use minimum processors
clf = LinearRegression(n_jobs=-1)

# to fit our training set
clf.fit(X_train, y_train)

# to see out model's accuracy on testing data
accuracy = clf.score(X_test, y_test)

print(accuracy)

'''
# here we will get out predicted value
'''
# forecast_set = clf.predict(X)
# print(forecast_set, accuracy, forecast_out)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)



# df['cost_without_service_tax'].plot()
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Cost without Service Tax')
# plt.show()
