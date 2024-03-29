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
from collections import defaultdict

sensitivity = 0
false_rate = 0
specificity = 0
# Defining global variables
epochs = 1
algorithms = ["SVM", "KNN", "Logistic Regression", "Naive Bayes", "Random Forest"]

# Defining data loading function for single thread execution

file1 = "Features.csv"
file_counts = defaultdict(int)
acts = 0

def _LoadData_SingleThread(array):
    data_time_s = time.time()
    dataset = pd.read_csv(file1, header=None)
    dataset.dropna(axis=0, inplace=True)
    entries = dataset.iloc[:, -1].values
    activity = dataset.iloc[:, -2].values
    print("Activity Length")
    print(len(activity))
    global acts
    acts = len(activity)

    activity_index = []
    test_activity = []

    for i in range(len(entries)):
        activity_index.append(i)
    activity_index = pd.Series(activity_index)
    dataset.insert(len(dataset.columns), len(dataset.columns), activity_index.values)

    X = dataset.iloc[:, :-3].values
    X = pd.DataFrame(X)
    X.insert(len(X.columns), len(X.columns), dataset.iloc[:, -1].values)

    # X = dataset;
    y = dataset.iloc[:, -2].values

    X_train_a, X_test_a, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    X_train_a = pd.DataFrame(X_train_a)
    X_test_a = pd.DataFrame(X_test_a)
    test_activity_index = X_test_a.iloc[:, -1].values
    for index in test_activity_index:
        test_activity.append(activity[int(index)])
    # print(test_activity)
    # print(y_test)

    X_train = X_train_a.iloc[:, :-1].values
    X_test = X_test_a.iloc[:, :-1].values

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    data_time_e = time.time()
    data_time = data_time_e - data_time_s
    print("Data Loading time: " + str(data_time))
    # array.append(data_time)
    print("Test Activity Length")
    print(len(test_activity))
    return X_train, X_test, y_train, y_test, test_activity


# Defining ML training function for single thread execution
def _TrainModel_SingleThread(X_train, X_test, y_train, y_test, ModelName, array):
    training_time_s = time.time()
    if ModelName == "SVM":
        classifier = SVC(C=10, kernel="rbf", random_state=0, probability=True)
        classifier.fit(X_train, y_train)
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
    training_time_e = time.time()
    training_time = training_time_e - training_time_s
    print("Training time: " + str(training_time))
    # array.append(training_time)
    return classifier


# Defining ML testing function for single thread execution
def _TestModel_SingleThread(classifier, array, test_activity):
    testing_time_s = time.time()
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    sensitivity = tp / (tp + fn)
    #false_rate = fn / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    testing_time_e = time.time()
    testing_time = testing_time_e - testing_time_s
    print("Testing time: " + str(testing_time))
    print("Acc " + str(accuracy))
    print("Sens " + str(sensitivity))
    #print("fals " + str(false_rate))
    print("Specs " + str(specificity))
    # array.append(testing_time)
    array.append(accuracy)
    array.append(sensitivity)
    #array.append(false_rate)
    array.append(specificity)

    for i in range(len(y_pred)):
        if y_test[i] != y_pred[i]:
            # array.append(test_activity[i])
            file_counts[test_activity[i]] += 1


f = open((f"{file1.split('.')[0]}_ensemble_results.csv"), "w", newline="")
writer = csv.writer(f)
writer.writerow(["Algorithm", "Accuracy", "Sensitivity", "Specificity"])
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


fails = []
failed_falls = 0
failed_adls = 0

for fil, c in file_counts.items():
    if c >= 3:
        fails.append(fil)
        if "adl" in fil: failed_adls += 1
        if "fall" in fil: failed_falls += 1

writer.writerow([])
writer.writerow(["Combinations", "Files failed",
                "Failed Activities #", "Failed Falls", "Failed ADLs", "Total Features",
                "Total Falls", "Total ADLs", "Ensemble Accuracy", "Ensemble Sensitivity", "Ensemble Specificity"])

row = []
row.append(file1.split('.')[0])
row.append("\r\n".join(fails))

failed_acts = failed_adls + failed_falls
row += [failed_acts, failed_falls, failed_adls]

falls = acts//3
adls = acts//1.5
row += [acts, falls, adls]
row += [1-failed_acts/acts, 1-failed_falls/falls, 1-failed_adls/adls]
writer.writerow(row)

f.close()
