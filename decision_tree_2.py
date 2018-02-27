

from sklearn.datasets import fetch_mldata
from sklearn import tree
from sklearn import metrics
from sklearn import cross_validation
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import io
import pydot
# import preprocessing as prep

mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

print(X)

print("starting decision tree")
depth = 64
crf = tree.DecisionTreeClassifier(criterion="gini", max_depth=depth, max_features=784)
print(crf)
crf = crf.fit(X[:1000], y[:1000])
predictionRes = crf.predict(X)
print("finished decision tree")

# print(metrics.classification_report(X.tolist(), predictionRes, digits=4))

# Cross Validation Results Exercise 3.3 for Decision Tree
scores = cross_validation.cross_val_score(crf, X[:1000], y[:1000].tolist(), cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

# Pixel importances on 28*28 image
importances = crf.feature_importances_
importances = importances.reshape((28, 28))

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances for decision tree, Depth = %d, accuracy = %d" % (depth, scores.mean()*100))
plt.show()

# Decision Tree as output -> decision_tree.png
dot_data = io.StringIO()
tree.export_graphviz(crf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('decision_tree.png')
