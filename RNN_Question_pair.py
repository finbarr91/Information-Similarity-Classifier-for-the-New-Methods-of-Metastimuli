# Time series forecasting with LSTM Neural network Python
import tensorflow as tf
import keras
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import quandl
import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras import regularizers
from keras.regularizers import Regularizer
from keras import Sequential
from keras.layers import Dense, LSTM
from keras import backend as k
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn import model_selection
import math

df = pd.read_csv('df_fe_without_preprocessing_train.csv')
print(df.head())

X = np.array(df.drop(['id','qid1','qid2','question1','question2','is_duplicate'],1))
predict = df['is_duplicate']
y = np.array(predict)

# Normalizing the data:
scaler = StandardScaler()
X = scaler.fit_transform(X)



# Training the Model
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)

print('shape of train data:',X_train.shape)
print(y_train.shape)


model = keras.Sequential()
model.add(Dense(40, input_shape =(11,), activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(1,activation='linear'))

# Summary of model
print(model.summary())

model.compile(optimizer= 'adam', loss= 'mean_squared_error', metrics=['accuracy'])

# Fitting the model
model_hist = model.fit(X_train,y_train,epochs=5,verbose=2,validation_split=0.2, batch_size=25)

# Evaluating the model
eval = model.evaluate(X_test,y_test)
print(eval)

print(model_hist.history.keys())

# Visualizing the accuracies and losses of the model
fig = plt.figure()

plt.subplot(2,2,1)
plt.plot(model_hist.history['loss'])
plt.title('Loss graph')
plt.xlabel('Number of epochs')
plt.ylabel('Model loss')

plt.subplot(2,2,2)
plt.plot(model_hist.history['val_loss'])
plt.title('validation loss graph')
plt.xlabel('Number of epochs')
plt.ylabel('Validation loss')


plt.subplot(2,2,3)
plt.plot(model_hist.history['accuracy'])
plt.title('accuracy graph')
plt.xlabel('Number of epochs')
plt.ylabel('Model\'s accuarcy')


plt.subplot(2,2,4)
plt.plot(model_hist.history['val_accuracy'])
plt.title('val_accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Model\'s validation accuracy')


plt.tight_layout(pad=2)

plt.show()

# To make predictions
y_pred = model.predict(X_test)
y_pred = abs(np.argmax(y_pred))
for i in range(len(y_pred)):
    print('The expected purchase amount is', y_pred, 'the features used is making predictions are:',X_test)

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

plot_confusion_matrix(y_test,y_pred) # Calling the plot confusion matrix
Print_Model_performance(y_test,y_pred)

accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
print('Logistic Accuracy: %0.3f (+/- %0.3f)' % (accuracies.mean(), accuracies.std() * 2))