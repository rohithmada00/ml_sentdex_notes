import quandl , math
import pandas as pd
import numpy as np 
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

# What is a data frame class 
# What is a feature ?
# What is a label ?
# Why is linear regression threaded ? Is SVM threaded ?
# Preprocessing ? Fit ? Score ?


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. High','Adj. Low','Adj. Open','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. High']*100
df['OC_PCT'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Close']*100

df = df[['Adj. Close','HL_PCT','OC_PCT','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# print(df.tail()) 

# X - feature , y - label

X = np.array(df.drop(['label'] , axis = 1))
y = np.array(df['label'])
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)

# Using Linear regression algo

clf = LinearRegression().fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)  

print(accuracy)
 
# Using SVM

clf = svm.SVR(kernel = 'poly').fit(X_train, y_train) 
accuracy = clf.score(X_test,y_test)  

print(accuracy)
