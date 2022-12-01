import threading
import pandas as pd
import random
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

file = "Features_Entire_Dataset_with_heart_rate.csv"
# file = "Features_Entire_Dataset_without_heart_rate.csv"
dataset = pd.read_csv(file, header=None)

dataset.dropna(axis=0, inplace=True)

users_to_drop = random.sample(range(6, 42), 11)

print(users_to_drop)

for user in users_to_drop:
    dataset = dataset[~dataset[112].str.contains(f"user{user}_")]
    # dataset = dataset[~dataset[105].str.contains(f"user{user}_")]


entries = dataset.iloc[:, -1].values
activity = dataset.iloc[:, -2].values
print("Activity Length")
print(len(activity))

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

users = len(y) // 24


# Defining data loading function for single thread execution
def _LoadData_SingleThread():
    # data_time_s = time.time()
    #
    print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
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
    #
    # data_time_e = time.time()
    # data_time = data_time_e - data_time_s
    # print("Data Loading time: " + str(data_time))

    return X_train, X_test, y_train, y_test


# Defining ML training function for single thread execution
def _TrainModel_SingleThread(X_train, X_test, y_train, y_test, ModelName):
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
    # training_time_e = time.time()
    # training_time = training_time_e - training_time_s
    # print("Training time: " + str(training_time))

    return classifier


# Defining ML testing function for single thread execution
def _TestModel_SingleThread(classifier, array):
    testing_time_s = time.time()
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    # fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    sensitivity = tp / (tp + fn)
    false_rate = fn / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Sensitivity: {sensitivity}")
    print(f"False rate: {false_rate}")
    print(f"Specificity: {specificity}")
    # print(cm)
    print(f"Accuracy: {accuracy}")
    # testing_time_e = time.time()
    # testing_time = testing_time_e - testing_time_s
    # print("Testing time: " + str(testing_time))
    array.append(accuracy)
    array.append(sensitivity)
    array.append(false_rate)
    array.append(specificity)

    for i in range(len(y_pred)):
        if y_test[i] != y_pred[i]:
            array.append(test_activity[i])


f = open(f"Results_{file.split('.')[0]}_using {users} users.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["Algorithm", "Accuracy", "Sensitivity", "False Rate", "Specificity"])
for algo in algorithms:
    print(algo)
    for u in tqdm(range(epochs)):
        # loadData
        array = []
        X_train, X_test, y_train, y_test = _LoadData_SingleThread()
        # trainModel
        classifier = _TrainModel_SingleThread(X_train, X_test, y_train, y_test, algo)
        # testModel
        _TestModel_SingleThread(classifier, array)
        writer.writerow([algo] + array)
f.close()
