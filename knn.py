from preprocessing import load_data
import numpy as np

import pandas as pd
from pandas import Series, DataFrame

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image
# from IPython import get_ipython
# get_ipython().magic('matplotlib', 'inline')

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

print(validate_labels)
np.random.seed(666)
train_data, train_labels = shuffle(train_data, train_labels)
validate_data, validate_labels = shuffle(validate_data, validate_labels)
test_data, test_labels = shuffle(test_data, test_labels)

#This function displays one or more images in a grid manner.
def show_img_with_neighbors(imgs, n=1):                       
  fig = plt.figure()                                          
  for i in range(0, n):                                      
      fig.add_subplot(1, n, i, xticklabels=[], yticklabels=[])
      if n == 1:                                              
          img = imgs                                          
      else:                                                   
          img = imgs[i]                                       
      plt.imshow(img.reshape((28, 28)), cmap="Greys")           

#This function shows some images for which k-NN made a mistake
# For each of the missed image, it will also show k most similar images so that you will get an idea of why it failed. 
def show_erroring_images_for_model(errors_in_model, num_img_to_print, model, n_neighbors): 
  for errorImgIndex in errors_in_model[:num_img_to_print]:                             
      error_image = test_data[errorImgIndex].reshape((28,28))                           
      not_needed, result = model.kneighbors(error_image, n_neighbors=n_neighbors)      
      show_img_with_neighbors(error_image)                                             
      show_img_with_neighbors(train_data[result[0],:], len(result[0]))

#Step 1: Create a classifier with appropriate parameters
knn_model_k1 = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
# #Step 2: Fit it with training data
knn_model_k1.fit(train_data, train_labels)
# #Print the model so that you know all parameters
print(knn_model_k1)                             
# #Step 3: Make predictions based on testing data
predictions_knn_model_k1 = knn_model_k1.predict(validate_data)
# #Step 4: Evaluate the data
print("Accuracy of K-NN with k=1 is", metrics.accuracy_score(validate_labels, predictions_knn_model_k1))  

knn_model_k3 = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
knn_model_k3.fit(train_data, train_labels)
print (train_labels[0:500])
predictions_knn_model_k3 = knn_model_k3.predict(validate_data)
print (predictions_knn_model_k3[0:500])                                              
print ("Accuracy of K-NN with k=3 is", metrics.accuracy_score(validate_labels, predictions_knn_model_k3))

knn_model_k5 = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
knn_model_k5.fit(train_data, train_labels)
predictions_knn_model_k5 = knn_model_k5.predict(validate_data)                                               
print ("Accuracy of K-NN with k=5 is", metrics.accuracy_score(validate_labels, predictions_knn_model_k5))
#Let us now see the first five images that were predicted incorrectly see what the issue is                                  
# errors_knn_model_k1 = [i for i in range(0, len(test_images)) if predictions_knn_model_k1[i] != test_labels[i]]
# show_erroring_images_for_model(errors_knn_model_k1, 5, knn_model_k1, 1)