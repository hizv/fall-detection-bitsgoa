import threading
import pandas as pd
import numpy
import time
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

sensitivity = 0
false_rate = 0
specificity = 0
# Defining global variables
epochs = 1
algorithms = ["SVM", "KNN", "Logistic Regression", "Naive Bayes", "Random Forest"]

file1 = "Features_Age_under30.csv"  # Training
file2 = "Features_Age_30to40.csv"  # Testing

# Defining data loading function for single thread execution
def _LoadData_SingleThread(array):
    # data_time_s = time.time()
    dataset = pd.read_csv(file1, header=None)
    dataset.dropna(axis=0, inplace=True)

    entries = dataset.iloc[:, -1].values
    activity = dataset.iloc[:, -2].values
    print("Activity Length")
    print(len(activity))

    activity_index = []
    test_activity = []

    for i in range(len(entries)):
        activity_index.append(i)
    activity_index = pd.Series(activity_index)

    X_train = dataset.iloc[:, :-2].values
    y_train = dataset.iloc[:, -1].values

    dataset = pd.read_csv(file2, header=None)
    dataset.dropna(axis=0, inplace=True)
    X_test = dataset.iloc[:, :-2].values
    y_test = dataset.iloc[:, -1].values
    test_activity = dataset.iloc[:, -2].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # data_time_e = time.time()
    # data_time = data_time_e - data_time_s
    # print("Data Loading time: " + str(data_time))
    # array.append(data_time)
    return X_train, X_test, y_train, y_test, test_activity


# Defining ML training function for single thread execution
def _TrainModel_SingleThread(X_train, X_test, y_train, y_test, ModelName, array):
    # training_time_s = time.time()
    if ModelName == "SVM":
        print("ok")
        classifier = SVC(C=10, kernel="rbf", random_state=0, probability=True)
        print("ok")
        classifier.fit(X_train, y_train)
        print("ok")
        scores = cross_val_score(classifier, X_train, y_train, cv=10)

    elif ModelName == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=9)
        classifier.fit(X_train, y_train)
        scores = cross_val_score(classifier, X_train, y_train, cv=10)

    elif ModelName == "Logistic Regression":
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        scores = cross_val_score(classifier, X_train, y_train, cv=10)

    elif ModelName == "Naive Bayes":
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        scores = cross_val_score(classifier, X_train, y_train, cv=10)

    elif ModelName == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=170)
        classifier.fit(X_train, y_train)
        scores = cross_val_score(classifier, X_train, y_train, cv=10)
    # training_time_e = time.time()
    # training_time = training_time_e - training_time_s
    # print("Training time: " + str(training_time))
    # array.append(training_time)
    return classifier


# Defining ML testing function for single thread execution
def _TestModel_SingleThread(classifier, array, test_activity):
    # testing_time_s = time.time()
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # print("----------------")
    # print(y_test)
    # print("----------------")
    # print(y_pred)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    sensitivity = tp / (tp + fn)
    false_rate = fn / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    # testing_time_e = time.time()
    # testing_time = testing_time_e - testing_time_s
    # print("Testing time: " + str(testing_time))
    print("Accuracy " + str(accuracy))
    print("Sensitivity " + str(sensitivity))
    print("False rate " + str(false_rate))
    print("Specificity " + str(specificity))

    # array.append(testing_time)
    array.append(accuracy)
    array.append(sensitivity)
    array.append(false_rate)
    array.append(specificity)

    for i in range(len(y_pred)):
        if y_test[i] != y_pred[i]:
            array.append(test_activity[i])


f = open((f"{file1.split('.')[0]}_{file2.split('.')[0]}.csv"), "w", newline="")
writer = csv.writer(f)
writer.writerow(["Algorithm", "Accuracy", "Sensitivity", "False Rate", "Specificity"])
for algo in algorithms:
    print(algo)
    for u in tqdm(range(epochs)):
        # loadData
        array = []
        X_train, X_test, y_train, y_test, test_activity = _LoadData_SingleThread(array)
        # trainModel
        classifier = _TrainModel_SingleThread(
            X_train, X_test, y_train, y_test, algo, array
        )
        # testModel
        _TestModel_SingleThread(classifier, array, test_activity)
        writer.writerow([algo] + array)
f.close()
