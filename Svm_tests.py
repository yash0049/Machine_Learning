from sklearn import datasets
import numpy
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

#split dataset in features and labels
X = iris.data
y = iris.target

# pyplot.plot(X,y)
# print(X.shape)
# print(y.shape)

#We need to split data into test data and train data to check the accuracy of the model
#Here we are splitting into 80-20 so 80% to train and 20% data to test the accuracy of model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#Train svm model

model = svm.SVC()
model.fit(X_train,y_train)

Prediction = model.predict(X_test)
Accuracy = accuracy_score(y_test,Prediction)

print("Prediction: ",Prediction)
print("Accuracy: ", Accuracy)
