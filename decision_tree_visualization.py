import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

#load datasets
iris = load_iris()

# print (iris.feature_names)
# print (iris.target_names)
# print (iris.data[0])
# print (iris.target[0])


# for i in range(len(iris.target)):
#     print ("Example %d : %s -- %s" % (i, iris.target[i],iris.data[i]))

#training datasets
train_idx = [0,50,100]
train_target = np.delete(iris.target, train_idx)
train_data = np.delete(iris.data, train_idx, axis = 0)

#testing datasets
test_target = iris.target[train_idx]
test_data = iris.data[train_idx]

#train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target)

#print the output of testData
print (test_target)

#print the prediction from classifier
print(clf.predict(test_data))

#print the visualization onto PDF
# from sklearn.externals.six import StringIO
# import pydot
# dot_data = StringIO()
#
# tree.export_graphviz(clf,out_file=dot_data,feature_names=iris.feature_names, class_names = iris.target_names, filled = True,rounded = True, impurity = False)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")
