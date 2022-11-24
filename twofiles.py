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

file1 = "Features_trial_under30.csv"
file2 = "Features_trial_30to40.csv"

# Defining data loading function for single thread execution
def _LoadData_SingleThread(array):
    data_time_s = time.time()
    dataset = pd.read_csv(file1)
    dataset.dropna(axis=0, inplace=True)
    X_train = dataset.iloc[:, :-1].values
    y_train = dataset.iloc[:, -1].values

    dataset = pd.read_csv(file2)
    dataset.dropna(axis=0, inplace=True)
    X_test = dataset.iloc[:, :-1].values
    y_test = dataset.iloc[:, -1].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    data_time_e = time.time()
    data_time = data_time_e - data_time_s
    # print("Data Loading time: " + str(data_time))
    # array.append(data_time)
    return X_train, X_test, y_train, y_test


# Defining ML training function for single thread execution
def _TrainModel_SingleThread(X_train, X_test, y_train, y_test, ModelName, array):
    training_time_s = time.time()
    if ModelName == "SVM":
        print("ok")
        classifier = SVC(C=10, kernel="rbf", random_state=0, probability=True)
        print("ok")
        classifier.fit(X_train, y_train)
        print("ok")
        scores = cross_val_score(classifier, X_train, y_train, cv=10)

    elif ModelName == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=30)
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
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        scores = cross_val_score(classifier, X_train, y_train, cv=10)
    training_time_e = time.time()
    training_time = training_time_e - training_time_s
    print("Training time: " + str(training_time))
    # array.append(training_time)
    return classifier


# Defining ML testing function for single thread execution
def _TestModel_SingleThread(classifier, array):
    testing_time_s = time.time()
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    sensitivity = tp / (tp + tn)
    false_rate = fn / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    testing_time_e = time.time()
    testing_time = testing_time_e - testing_time_s
    print("Testing time: " + str(testing_time))
    print("Acc " + str(accuracy))
    print("Sens " + str(sensitivity))
    print("fals " + str(false_rate))
    print("Specs " + str(specificity))
    # array.append(testing_time)
    array.append(accuracy)
    array.append(sensitivity)
    array.append(false_rate)
    array.append(specificity)


f = open((f"{file1.split('.')[0]}_{file2.split('.')[0]}.csv"), "w")
writer = csv.writer(f)
writer.writerow(["Algorithm", "Accuracy", "Sensitivity", "False Rate", "Specificity"])
for algo in algorithms:
    print(algo)
    for u in tqdm(range(epochs)):
        # loadData
        array = []
        X_train, X_test, y_train, y_test = _LoadData_SingleThread(array)
        # trainModel
        classifier = _TrainModel_SingleThread(
            X_train, X_test, y_train, y_test, algo, array
        )
        # testModel
        _TestModel_SingleThread(classifier, array)
        writer.writerow([algo] + array)
f.close()
