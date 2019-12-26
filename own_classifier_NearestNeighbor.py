from scipy.spatial import distance

def eucDistance(a,b):
    return distance.euclidean(a,b)

class myKNN():
    def fit(self, x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self,row):
        best_distance = eucDistance(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = eucDistance(row,self.x_train[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i

        return self.y_train[best_index]


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
# from sklearn.neighbors import KNeighborsClassifier
# myClassifier = KNeighborsClassifier()
myClassifier = myKNN()

myClassifier.fit(x_train, y_train)
predictions = myClassifier.predict(x_test)

print (predictions)
print ( y_test)

from sklearn.metrics import accuracy_score
print ( accuracy_score(y_test, predictions) )