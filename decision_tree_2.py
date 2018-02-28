

from sklearn.datasets import fetch_mldata
from sklearn import tree
# from sklearn import metrics
from sklearn import cross_validation
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import numpy as np
import io
import pydot
import pandas as pd
# import preprocessing as prep

mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))


print("starting decision tree")
depth = 64
generated_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth, max_features=784)
print(generated_tree)
generated_tree = generated_tree.fit(X[:1000], y[:1000])
prediction = generated_tree.predict(X)
print("finished decision tree")

print(prediction)

# print(metrics.classification_report(X.tolist(), predictionRes, digits=4))

# Cross Validation Results Exercise 3.3 for Decision Tree
scores = cross_validation.cross_val_score(generated_tree, X[:1000], y[:1000].tolist(), cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

# Pixel importances on 28*28 image
importances = generated_tree.feature_importances_
importances = importances.reshape((28, 28))

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances for decision tree, Depth = %d, accuracy = %d" % (depth, scores.mean()*100))
plt.show()

# Decision Tree as output -> decision_tree.png
dot_data = io.StringIO()
tree.export_graphviz(generated_tree, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('decision_tree.png')

# print(y)

y_test_matrix = y #[np.where(r == 1)[0][0] for r in y]
y_prediction_matrix = [i for i in prediction.tolist()]

count = {0: [0, -1], 1: [0, -1], 2: [0, -1], 3: [0, -1], 4: [0, -1], 5: [0, -1], 6: [0, -1], 7: [0, -1], 8: [0, -1], 9: [0, -1]}
correct_preds=0
for i in range(len(y)):
    if y_test_matrix[i] == y_prediction_matrix[i]:
        correct_preds += 1
    else:
        count[y_prediction_matrix[i]] = [count[y_prediction_matrix[i]][0]+1, i]

count = sorted(count.items(), key=lambda x: x[1][0], reverse=True)


y_test_matrix = pd.Series(y_test_matrix, name='Actual')
y_prediction_matrix = pd.Series(y_prediction_matrix, name='Predicted')
df_confusion = pd.crosstab(y_test_matrix, y_prediction_matrix, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('\n', df_confusion)
