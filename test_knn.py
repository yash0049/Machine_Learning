import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('datasets/car.data')
# print(data.head())

X = data[['buying','maint','safety']].values
y = data[['class']]

# print(X, y)

#We need to convert string features to numeric value for model to process it
#we use label encoder for that

Le = LabelEncoder()
# print('hi')
# print(X[0])
# print(X[:, 1])

for i in range(len(X[0])):
    X[:, i]= Le.fit_transform(X[:, i])

# print(X)  

#transforming y values using label_mapping
label_mapping = {'unacc':0, 'acc':1, 'good':2, 'vgood':3}
# print(y['class'])
y['class'] = y['class'].map(label_mapping)
y = np.array(y)
# print(y)

#Creating model

Knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

#Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

Knn.fit(X_train, y_train)

Prediction = Knn.predict(X_test)

Accuracy = metrics.accuracy_score(y_test, Prediction)

print('Prediction: ',Prediction)
print('Accuracy: ' , Accuracy)

a=32
print("Actual Value", y[a])
print("Predicted Value:", Knn.predict(X)[a])