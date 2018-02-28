from preprocessing import load_data
import numpy as np

import pandas as pd
from pandas import Series, DataFrame


import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata, load_boston                                                                       
from sklearn.utils import shuffle                                                                                            
from sklearn.neighbors import KNeighborsClassifier                                                                           
from sklearn import metrics                                                                                                  
from sklearn import tree                                                                                                     
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor                                                       
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB                                                       
from sklearn.svm import SVC, LinearSVC , SVR                                                                                 
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression                                            
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor                                                    
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier                                                       
from sklearn.cross_validation import KFold, train_test_split, cross_val_score                                                
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.utils import check_random_state

import pydot

mnist = fetch_mldata('MNIST Original')

labels = np.load('labels.npy')
data = np.load('images.npy')

train_data, train_labels, validate_data, validate_labels, test_data, test_labels  = [],[],[],[],[],[]

for i in range (0,9):
    index = 0
    num_data = []
    num_labels = []
    for num in labels:
        if num == i:
            num_data.extend(data[index])
            num_labels.append(i)
        index += 1

    length = len(num_data)
    len1 = int(0.6*length)
    len2 = int(0.75*length)

    train_data.extend(num_data[:len1])
    validate_data.extend(num_data[len1:len2])
    test_data.extend(num_data[len2:])

    train_labels.extend(np.repeat(i, len1))
    validate_labels.extend(np.repeat(i, (len2 - len1)))
    test_labels.extend(np.repeat(i, (length - len2)))

# for i in range(0,9):
#     num_data = mnist.data[mnist.target==i]
#     length = num_data.shape[0]
#     len1 = int(0.6*length)
#     len2 = int(0.75*length)

#     train_data.extend(num_data[:len1])
#     validate_data.extend(num_data[len1:len2])
#     test_data.extend(num_data[len2:])

#     train_labels.extend(np.repeat(i, len1))
#     # train_labels = np.hstack([np.repeat(i, train_data.shape[0]), train_labels])
#     validate_labels.extend(np.repeat(i, (len2 - len1)))
#     test_labels.extend(np.repeat(i, (length - len2)))

np.random.seed(666)
train_data, train_labels = shuffle(train_data, train_labels)
validate_data, validate_labels = shuffle(validate_data, validate_labels)
test_data, test_labels = shuffle(test_data, test_labels)

knn_model_k1 = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
knn_model_k1.fit(train_data, train_labels)                         
predictions_knn_model_k1 = knn_model_k1.predict(validate_data)
print("Accuracy of K-NN with k=1 is", metrics.accuracy_score(validate_labels, predictions_knn_model_k1))

test_matrix = validate_labels
prediction_matrix = [i for i in predictions_knn_model_k1.tolist()]

count = {0: [0, -1], 1: [0, -1], 2: [0, -1], 3: [0, -1], 4: [0, -1], 5: [0, -1], 6: [0, -1], 7: [0, -1], 8: [0, -1], 9: [0, -1]}
correct_preds=0
for i in range(len(test_matrix)):
    if test_matrix[i] == prediction_matrix[i]:
        correct_preds += 1
    else:
        count[prediction_matrix[i]] = [count[prediction_matrix[i]][0]+1, i]

count = sorted(count.items(), key=lambda x: x[1][0], reverse=True)

test_matrix = pd.Series(test_matrix, name='Actual')
prediction_matrix = pd.Series(prediction_matrix, name='Predicted')
df_confusion = pd.crosstab(test_matrix, prediction_matrix, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('\n', df_confusion)  

knn_model_k3 = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
knn_model_k3.fit(train_data, train_labels)
predictions_knn_model_k3 = knn_model_k3.predict(validate_data)                                           
print ("Accuracy of K-NN with k=3 is", metrics.accuracy_score(validate_labels, predictions_knn_model_k3))

test_matrix = validate_labels
prediction_matrix = [i for i in predictions_knn_model_k3.tolist()]

count = {0: [0, -1], 1: [0, -1], 2: [0, -1], 3: [0, -1], 4: [0, -1], 5: [0, -1], 6: [0, -1], 7: [0, -1], 8: [0, -1], 9: [0, -1]}
correct_preds=0
for i in range(len(test_matrix)):
    if test_matrix[i] == prediction_matrix[i]:
        correct_preds += 1
    else:
        count[prediction_matrix[i]] = [count[prediction_matrix[i]][0]+1, i]

count = sorted(count.items(), key=lambda x: x[1][0], reverse=True)

test_matrix = pd.Series(test_matrix, name='Actual')
prediction_matrix = pd.Series(prediction_matrix, name='Predicted')
df_confusion = pd.crosstab(test_matrix, prediction_matrix, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('\n', df_confusion) 

knn_model_k5 = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
knn_model_k5.fit(train_data, train_labels)
predictions_knn_model_k5 = knn_model_k5.predict(validate_data)                                               
print ("Accuracy of K-NN with k=5 is", metrics.accuracy_score(validate_labels, predictions_knn_model_k5))

test_matrix = validate_labels
prediction_matrix = [i for i in predictions_knn_model_k5.tolist()]

count = {0: [0, -1], 1: [0, -1], 2: [0, -1], 3: [0, -1], 4: [0, -1], 5: [0, -1], 6: [0, -1], 7: [0, -1], 8: [0, -1], 9: [0, -1]}
correct_preds=0
for i in range(len(test_matrix)):
    if test_matrix[i] == prediction_matrix[i]:
        correct_preds += 1
    else:
        count[prediction_matrix[i]] = [count[prediction_matrix[i]][0]+1, i]

count = sorted(count.items(), key=lambda x: x[1][0], reverse=True)

test_matrix = pd.Series(test_matrix, name='Actual')
prediction_matrix = pd.Series(prediction_matrix, name='Predicted')
df_confusion = pd.crosstab(test_matrix, prediction_matrix, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('\n', df_confusion) 