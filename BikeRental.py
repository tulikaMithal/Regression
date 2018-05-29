# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:42:56 2018

@author: lenovo
"""
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import itertools
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold




def plot_confusion_matrix(cm, classes, ImageFileName,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(ImageFileName)
    plt.close()
    

    
def PolyRegression(train_x_data, train_y_data, test_x_data, test_y_data):

    poly = PolynomialFeatures(degree=3)
    train_x_poly = poly.fit_transform(train_x_data)
    test_x_poly = poly.fit_transform(test_x_data)
    regrPoly = linear_model.Lasso(alpha = 1, tol=1)
    regrPoly.fit(train_x_poly, train_y_data)
    predictions = regrPoly.predict(test_x_poly)
    print("\nMean squared error using Linear Regression on test data:\n %.2f"
      % mean_squared_error(test_y_data, predictions))
    
def KNNRegression(train_x_data, train_y_data, test_x_data, test_y_data):
    
    knn = neighbors.KNeighborsRegressor(5, weights='distance')
    knn.fit(train_x_data, train_y_data)
    predictions = knn.predict(test_x_data)
    print("\nMean squared error using KNN Regression on test data:\n %.2f"
      % mean_squared_error(test_y_data, predictions))
    
def NeuralNetwork(train_x_data, train_y_data, test_x_data, test_y_data):
    mlp = MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter = 500)
    mlp.fit(train_x_data,train_y_data)
    predictions = mlp.predict(test_x_data)
    print("\nMean squared error using Neural Network on test data:\n %.2f"
      % mean_squared_error(test_y_data, predictions))
  
    
def CrossValidPolyRegression(train_x_data, train_y_data, test_x_data, test_y_data):
        
    train_xcv_data = train_x_data
    train_ycv_data = train_y_data
    
    kf = KFold(n_splits=5, shuffle = True)
    kf.get_n_splits(train_xcv_data)
    
    MSE = 0
    
    for train_index, test_index in kf.split(train_xcv_data):
        poly = PolynomialFeatures(degree=3)
        train_x_poly = poly.fit_transform(train_xcv_data.loc[train_index])
        test_x_poly = poly.fit_transform(train_xcv_data.loc[test_index])
        regrPoly = linear_model.Lasso(alpha = 1, tol = 1)
        regrPoly.fit(train_x_poly, train_ycv_data.loc[train_index])
        predictions = regrPoly.predict(test_x_poly)
        MSE += mean_squared_error(train_ycv_data.loc[test_index], predictions)
        
    print("\nMean Square Error (MSE) using Linear Regression and 5-fold cross validation :\n", MSE/5.0)
    
    
    
def CrossValidNeuralNetwork(train_x_data, train_y_data, test_x_data, test_y_data):
    
    train_xcv_data = train_x_data
    train_ycv_data = train_y_data
    
    kf = KFold(n_splits=5, shuffle = True)
    kf.get_n_splits(train_xcv_data)
    
    MSE = 0
    for train_index, test_index in kf.split(train_xcv_data):
        mlp = MLPRegressor(hidden_layer_sizes=(30,30,30),  max_iter = 500)
        mlp.fit(train_xcv_data.loc[train_index], train_ycv_data.loc[train_index])
        predictions = mlp.predict(train_xcv_data.loc[test_index])
        MSE += mean_squared_error(train_ycv_data.loc[test_index], predictions)
        
    print("\nMean Square Error (MSE) using Neural Network and 5-fold cross validation :\n", MSE/5.0)
    
    
def CrossValidKNNRegression(train_x_data, train_y_data, test_x_data, test_y_data):
    train_xcv_data = train_x_data
    train_ycv_data = train_y_data
    
    kf = KFold(n_splits=5, shuffle = True)
    kf.get_n_splits(train_xcv_data)
    
    MSE = 0
    for train_index, test_index in kf.split(train_xcv_data):
        knn = neighbors.KNeighborsRegressor(5, weights='distance')
        knn.fit(train_xcv_data.loc[train_index], train_ycv_data.loc[train_index])
        predictions = knn.predict(train_xcv_data.loc[test_index])
        MSE += mean_squared_error(train_ycv_data.loc[test_index], predictions)
        
    print("\nMean Square Error (MSE) using KNN and 5-fold cross validation :\n", MSE/5.0)
    
train_data = pd.read_csv(sys.argv[1])
test_data = pd.read_csv(sys.argv[2])

train_x_data = train_data.iloc[:,3:-3]
train_y_data = train_data.iloc[:,-1]

test_x_data = test_data.iloc[:,3:-3]
test_y_data = test_data.iloc[:,-1]

CrossValidPolyRegression(train_x_data, train_y_data, test_x_data, test_y_data)
PolyRegression(train_x_data, train_y_data, test_x_data, test_y_data)

CrossValidKNNRegression(train_x_data, train_y_data, test_x_data, test_y_data)
KNNRegression(train_x_data, train_y_data, test_x_data, test_y_data)

CrossValidNeuralNetwork(train_x_data, train_y_data, test_x_data, test_y_data)
NeuralNetwork(train_x_data, train_y_data, test_x_data, test_y_data)

