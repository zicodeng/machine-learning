import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

df = quandl.get('WIKI/GOOGL')
# Select columns
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Create a new column for percentage of high and low.
# The calculation operates on per row basis.
df['HL_Pct'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0

df['Pct_Change'] = (df['Adj. Close'] - df['Adj. Open']) / \
    df['Adj. Open'] * 100.0

# Features
df = df[['Adj. Close', 'HL_Pct', 'Pct_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

# If a cell has value of NA, replace it with a different value.
# If machine learning, we cannot have any NA data.
# We can either remove that data (but generally ignoring data is bad),
# or make that data become an outlier.
df.fillna(-999999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

# Shift index by desired number of periods.
# For example, if shift by -1, current index is 2004-08-19,
# label column will have value of forecast_col at index 2004-08-20.
df['label'] = df[forecast_col].shift(-forecast_out)

# Drop label column (because this is not a feature, it is outcome),
# df.drop will return a copy of dataframe,
# np.array then converts this copy to array.
x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x = x[:-forecast_out]  # features
# These data do not have outcome, and thus should not be used as training data.
# But once our model is trained, we can predict outcome for them.
x_lately = x[-forecast_out:]

df.dropna(inplace=True)

y = np.array(df['label'])  # outcome

# Shuffle data around and use 80% data for features, 20% data for test.
train_features, test_features, train_outcome, test_outcome = model_selection.train_test_split(
    x, y, test_size=0.2)

# n_jobs allows us to adjust the number of threads we want to use on trianing the model.
# -1: all available threads.
clf = LinearRegression(n_jobs=-1)  # Define classifier

# svm allows us to easily switch between different algorithms.
# clf = svm.SVR(kernel='poly') # Try different algorithm: support vector regression
clf.fit(train_features, train_outcome)  # Train model

accuracy = clf.score(test_features, test_outcome)  # Test model
print('ACCURACY', accuracy)

# We can pass single value or an array of values we want to predict.
forecast_set = clf.predict(x_lately)

print('PREDICTED DAYS', forecast_out)
print('PREDICTED OUTCOME', forecast_set)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
