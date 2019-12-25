from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train , x_test, y_train  , y_test = train_test_split(x , y, test_size = 0.5)

#DecisionTreeClassifier clasification
# from sklearn import tree
# myClassifier = tree.DecisionTreeClassifier()

#Neighbors classification
from sklearn.neighbors import KNeighborsClassifier
myClassifier = KNeighborsClassifier()

myClassifier.fit(x_train, y_train)
predictions = myClassifier.predict(x_test)

print (predictions)
print ( y_test)

from sklearn.metrics import accuracy_score
print ( accuracy_score(y_test, predictions) )
