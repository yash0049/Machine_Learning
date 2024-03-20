import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('datasets/car.data')
print(data.head())

X = data[['buying','maint','safety']].values
y = data[['class']]

print(X, y)

#We need to convert string features to numeric value for model to process it
#we use label encoder for that

Le = LabelEncoder()
print('hi')
print(X[0])
print(X[:, 1])

for i in range(len(X[0])):
    X[:, i]= Le.fit_transform(X[:, i])

print(X)    