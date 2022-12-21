import threading
import random
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

# file1 = "Features_Entire_Dataset_with_heart_rate.csv"
file1 = "Features_Entire_Dataset_with_heart_rate.csv"
file_counts = defaultdict(int)


data_time_s = time.time()
dataset = pd.read_csv(file1, header=None)
dataset.dropna(axis=0, inplace=True)


# users_to_drop = random.sample(range(6, 42), 11)

# print(users_to_drop)

# for user in users_to_drop:
#     # dataset = dataset[~dataset[112].str.contains(f"user{user}_")]
#     dataset = dataset[~dataset[105].str.contains(f"user{user}_")]


entries = dataset.iloc[:, -1].values
activity = dataset.iloc[:, -2].values
print("Activity Length")
print(len(activity))


activity_index = []
test_activity = []

for i in range(len(entries)):
    activity_index.append(i)
activity_index = pd.Series(activity_index)

X = dataset.iloc[:, :-2].values
X = pd.DataFrame(X)


# X = dataset;
y = dataset.iloc[:, -1].values

print(X, y)
users = len(y) // 24

dataset.to_csv(f"{file1} with {users} users.csv", index=False, header=False)


def _LoadData_SingleThread(array):

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
        classifier = LogisticRegression(max_iter=4000)
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

    for i in range(len(y_pred)):
        if y_test[i] != y_pred[i]:
            file_counts[test_activity[i]] += 1


f = open(
    f"Reduced_{file1.split('.')[0]}_using {users} users_latest.csv", "w", newline=""
)
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

fails = []
for fil, c in file_counts.items():
    if c >= 3:
        writer.writerow([fil])
f.close()
