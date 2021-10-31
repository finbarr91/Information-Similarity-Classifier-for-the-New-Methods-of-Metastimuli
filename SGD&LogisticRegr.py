import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import sqlite3
from sqlalchemy import create_engine # database connection
import csv
import os
warnings.filterwarnings("ignore")

import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import sklearn


pickle_in=open('infosimilarity_columns.pickle','rb')
saved_model1= pickle.load(pickle_in)
data = saved_model1
pickle_in=open('infosimilarity_y_true.pickle','rb')
saved_model2= pickle.load(pickle_in)
y_true = np.array(saved_model2)

print('\nsaved_model1\n', saved_model1)
print('\nsaved_model2\n', saved_model2)
# Training the Model
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data,y_true,stratify=y_true,test_size=0.3, shuffle=True)


print('shape of train data:',X_train.shape)
print(X_test.shape)



# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    cm = confusion_matrix(y_test, y_pred)
    pm = (cm / cm.sum(axis=0))
    rm = (((cm.T) / (cm.sum(axis=1))).T)

    plt.figure(figsize=(10, 5))

    labels = [0, 1]
    # representing the Confusion matrix in heatmap format

    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True,  fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")

    # representing the Precision matrix in heatmap format
    plt.subplot(1, 3, 2)
    sns.heatmap(pm, annot=True,  fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")

    plt.subplot(1, 3, 3)
    # representing the Recall matrix in heatmap format
    sns.heatmap(rm, annot=True,  fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")

    plt.tight_layout()
    plt.show()

def Print_Model_performance(y_test,y_pred):
    print("\nTest Data Accuracy : %0.4f\n" % accuracy_score(y_test, y_pred))
    print('\naccuracy score:\n', accuracy_score(y_test, y_pred))
    print('\n Precision score:\n', precision_score(y_test, y_pred))
    print('\nRecall score:\n', recall_score(y_test, y_pred))
    print('\n F1 score:\n', f1_score(y_test, y_pred))

# Using a Logistic Regression
model = LogisticRegression (random_state=0, penalty='l2')
best = 0
for _ in range(20):
    model = LogisticRegression (random_state=0, penalty='l2')
    model.fit(X_train,y_train)
    accuracy = model.score(X_test, y_test)
    print('accuracy:', accuracy)
    if accuracy > best:
        best = accuracy

y_pred = model.predict(X_test)

plot_confusion_matrix(y_test, y_pred) # Calling the plot function
Print_Model_performance(y_test,y_pred) # Calling the print statement for the model's performance

accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
print('Logistic Accuracy: %0.3f (+/- %0.3f)' % (accuracies.mean(), accuracies.std() * 2))

# Using the SGD Classifier
model  = SGDClassifier( penalty='l2', loss='log', random_state=42)
best = 0
for _ in range(20):
    model = LogisticRegression (random_state=0, penalty='l2')
    model.fit(X_train,y_train)
    accuracy = model.score(X_test, y_test)
    print('accuracy:', accuracy)
    if accuracy > best:
        best = accuracy

pred_y = model.predict(X_test)

plot_confusion_matrix(y_test, pred_y) # Calling the plot function

Print_Model_performance(y_test,y_pred) # Calling the print statement for the model's performance

accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
print('SGD Accuracy: %0.3f (+/- %0.3f)' % (accuracies.mean(), accuracies.std() * 2))

