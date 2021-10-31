import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import StandardScaler

pickle_in=open('infosimilarity_columns.pickle','rb')
saved_model1= pickle.load(pickle_in)
data = saved_model1
data = abs(data)
pickle_in=open('infosimilarity_y_true.pickle','rb')
saved_model2= pickle.load(pickle_in)
y_true = np.array(saved_model2)

print('\nsaved_model1\n', saved_model1)
print('\nsaved_model2\n', saved_model2)
# Training the Model
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data,y_true,stratify=y_true,test_size=0.3, shuffle=True)


print('shape of train data:',X_train.shape)
print(X_test.shape)


# USING MutinomialNB
model = MultinomialNB()
best = 0
for _ in range(20):
    model = MultinomialNB()
    model.fit(X_train,y_train)
    accuracy = model.score(X_test, y_test)
    print('accuracy:', accuracy)
    if accuracy > best:
        best = accuracy

y_pred = model.predict(X_test)

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

plot_confusion_matrix(y_test,y_pred) # Calling the plot confusion matrix
Print_Model_performance(y_test,y_pred)

accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
print('MNB Accuracy: %0.3f (+/- %0.3f)' % (accuracies.mean(), accuracies.std() * 2))
