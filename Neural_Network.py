import pandas as pd
import pickle
import os
import seaborn as sns
import itertools
from keras.losses import BinaryCrossentropy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dense,Dropout
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error,accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from keras.optimizers import Adam, SGD
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import pickle
import warnings
import sqlite3
from sqlalchemy import create_engine # database connection
import csv
import os
warnings.filterwarnings("ignore")
import datetime as dt


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


model = keras.Sequential()
model.add(Dense(40, input_shape =(794,), activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(1,activation='linear'))
# model.add(Dropout(0.3))

# Summary of model
print(model.summary())
model.compile(loss= 'mean_squared_error', optimizer = Adam(lr=0.001), metrics= ['accuracy'])

# Fitting the model
model_hist = model.fit(X_train,y_train,epochs=10,verbose=2,validation_split=0.2, batch_size=25, shuffle=True)
print('Shape of X_train', X_train.shape)
print('Shape of X_test', X_test.shape)
# Evaluating the training model
eval = model.evaluate(X_train,y_train)
print(eval)
print(model_hist.history.keys())


# Evaluating the testing model
eval = model.evaluate(X_test,y_test)
print(eval)

print(model_hist.history.keys())

# Visualizing the accuracies and losses of the model
fig = plt.figure()

plt.subplot(2,2,1)
plt.plot(model_hist.history['loss'], color='red')
plt.title('Loss graph')
plt.xlabel('Number of epochs')
plt.ylabel('Model loss')

plt.subplot(2,2,2)
plt.plot(model_hist.history['val_loss'], color='blue')
plt.title('validation loss graph')
plt.xlabel('Number of epochs')
plt.ylabel('Validation loss')


plt.subplot(2,2,3)
plt.plot(model_hist.history['accuracy'], color = 'green')
plt.title('accuracy graph')
plt.xlabel('Number of epochs')
plt.ylabel('Model\'s accuarcy')


plt.subplot(2,2,4)
plt.plot(model_hist.history['val_accuracy'], color = 'yellow')
plt.title('val_accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Model\'s validation accuracy')


plt.tight_layout(pad=2)

plt.show()

# To make predictions
y_pred = model.predict(X_test, batch_size=10,verbose=2)
y_pred = np.argmax(y_pred, axis= -1)
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

plot_confusion_matrix(y_test, y_pred)



# Choosing the SGD optimizer and Binary Crossentropy loss

model = keras.Sequential()
model.add(Dense(40, input_shape =(794,), activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
# model.add(Dropout(0.3))

# Summary of model
print(model.summary())
loss = tf.keras.losses.BinaryCrossentropy()
#model.compile(loss= 'mean_squared_error', optimizer = SGD(lr=0.001), metrics= ['accuracy'])
model.compile( loss = loss,optimizer = SGD(lr=0.001), metrics= ['accuracy'])

# Fitting the model
model_hist = model.fit(X_train,y_train,epochs=10,verbose=2,validation_split=0.2, batch_size=25, shuffle=True)
# Evaluating the training model
eval = model.evaluate(X_train,y_train)
print(eval)
print(model_hist.history.keys())


# Evaluating the testing model
eval = model.evaluate(X_test,y_test)
print(eval)

print(model_hist.history.keys())

# Visualizing the accuracies and losses of the model
fig = plt.figure()

plt.subplot(2,2,1)
plt.plot(model_hist.history['loss'], color='red')
plt.title('Loss graph')
plt.xlabel('Number of epochs')
plt.ylabel('Model loss')

plt.subplot(2,2,2)
plt.plot(model_hist.history['val_loss'], color='blue')
plt.title('validation loss graph')
plt.xlabel('Number of epochs')
plt.ylabel('Validation loss')


plt.subplot(2,2,3)
plt.plot(model_hist.history['accuracy'], color = 'green')
plt.title('accuracy graph')
plt.xlabel('Number of epochs')
plt.ylabel('Model\'s accuarcy')


plt.subplot(2,2,4)
plt.plot(model_hist.history['val_accuracy'], color = 'yellow')
plt.title('val_accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Model\'s validation accuracy')


plt.tight_layout(pad=2)

plt.show()

# To make predictions
y_pred = model.predict(X_test, batch_size=10,verbose=2)
y_pred = np.argmax(y_pred, axis= -1)

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

plot_confusion_matrix(y_test, y_pred)





