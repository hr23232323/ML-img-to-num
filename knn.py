from preprocessing import load_data
import numpy as np

import pandas as pd
from pandas import Series, DataFrame

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image

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

# import pydot, StringIO
itrain, ivalidate, itest, ltrain, lvalidate, ltest = load_data()

# print("Got here")
# mnist = fetch_mldata('MNIST Original')
# print("Fetched")
# X = mnist.data.astype('float64')
# y = mnist.target
# random_state = check_random_state(0)
# permutation = random_state.permutation(X.shape[0])
# X = X[permutation]
# y = y[permutation]
# X = X.reshape((X.shape[0], -1))

# print (X)
# print (y)
#This function displays one or more images in a grid manner.
def show_img_with_neighbors(imgs, n=1):                       
  fig = plt.figure()                                          
  for i in range(0, n):                                      
      fig.add_subplot(1, n, i, xticklabels=[], yticklabels=[])
      if n == 1:                                              
          img = imgs                                          
      else:                                                   
          img = imgs[i]                                       
      plt.imshow(img.reshape(28, 28), cmap="Greys")           

#This function shows some images for which k-NN made a mistake
# For each of the missed image, it will also show k most similar images so that you will get an idea of why it failed. 
def show_erroring_images_for_model(errors_in_model, num_img_to_print, model, n_neighbors): 
  for errorImgIndex in errors_in_model[:num_img_to_print]:                             
      error_image = itest[errorImgIndex]                           
      not_needed, result = model.kneighbors(error_image, n_neighbors=n_neighbors)      
      show_img_with_neighbors(error_image)                                             
      show_img_with_neighbors(itrain[result[0],:], len(result[0]))

#Step 1: Create a classifier with appropriate parameters
knn_model_k1 = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
#Step 2: Fit it with training data
knn_model_k1.fit(itrain, ltrain)
#Print the model so that you know all parameters
print(knn_model_k1)                             
#Step 3: Make predictions based on testing data
predictions_knn_model_k1 = knn_model_k1.predict(itest)
#Step 4: Evaluate the data
print("Accuracy of K-NN with k=1 is", metrics.accuracy_score(ltest, predictions_knn_model_k1))  

#Let us now see the first five images that were predicted incorrectly see what the issue is                                  
errors_knn_model_k1 = [i for i in range(0, len(itest)) if predictions_knn_model_k1[i] != ltest[i]]
show_erroring_images_for_model(errors_knn_model_k1, 5, knn_model_k1, 1)