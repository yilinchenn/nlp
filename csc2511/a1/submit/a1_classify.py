from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv


def data_serialize(clf_num, accuracy, recalls, precisions, confusion_mat):
    data = []
    data.append(clf_num)
    data.append(accuracy)
    data.extend(recalls)
    data.extend(precisions)
    for i in range(confusion_mat.shape[0]):
        data.extend(confusion_mat[i, :])
    return data


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    correct_num = 0
    total_num = 0
    for i in range(C.shape[0]):  # row
        for j in range(C.shape[1]):  # col
            if i == j:
                correct_num += C[i, j]
            total_num += C[i, j]
    return correct_num / total_num


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    # add along row
    recalls = []
    for i in range(C.shape[0]):  # row
        total = 0
        for j in range(C.shape[1]):  # col
            total += C[i, j]
        recalls.append(C[i, i] / total)
    return recalls


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    # add along column
    precisions = []
    for j in range(C.shape[1]):  # col
        total = 0
        for i in range(C.shape[0]):  # row
            total += C[i, j]
        precisions.append(C[j, j] / total)
    return precisions


def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('Section 3.1')
    feats = np.load(filename)
    X = feats['arr_0'][:, 0:173]
    Y = feats['arr_0'][:, 173]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    data = []  # a list of lists
    # return (X_train, X_test, y_train, y_test, 5)

    print("**************1. linear svc****************")
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    SVC_predict = clf.predict(X_test)
    SVC_confusion_mat = confusion_matrix(y_test, SVC_predict)
    # print(SVC_confusion_mat)
    acc1 = accuracy(SVC_confusion_mat)
    rec1 = recall(SVC_confusion_mat)
    prec1 = precision(SVC_confusion_mat)
    data1 = data_serialize(1, acc1, rec1, prec1, SVC_confusion_mat)
    data.append(data1)
    # print(data1)

    iBest = 1
    best_accuracy = acc1
    print("iBest = " + str(iBest) + "\n\n")

    print("**************2. radial svc gamma = 2****************")
    clf = SVC(gamma=2, max_iter = 1000)
    clf.fit(X_train, y_train)
    SVC_gamma_predict = clf.predict(X_test)
    SVC_gamma_confusion_mat = confusion_matrix(y_test, SVC_gamma_predict)
    #print(SVC_gamma_confusion_mat)
    acc2 = accuracy(SVC_gamma_confusion_mat)
    rec2 = recall(SVC_gamma_confusion_mat)
    prec2 = precision(SVC_gamma_confusion_mat)
    data2 = data_serialize(2, acc2, rec2, prec2, SVC_gamma_confusion_mat)
    data.append(data2)
    # print(data2)

    if acc2 > best_accuracy:
        iBest = 2
        best_accuracy = acc2
    print("iBest = " + str(iBest) + "\n\n")

    print("**************3. random forest****************")
    clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf.fit(X_train, y_train)
    forest_predict = clf.predict(X_test)
    forest_confusion_mat = confusion_matrix(y_test, forest_predict)
    #print(forest_confusion_mat)
    acc3 = accuracy(forest_confusion_mat)
    rec3 = recall(forest_confusion_mat)
    prec3 = precision(forest_confusion_mat)
    data3 = data_serialize(3, acc3, rec3, prec3, forest_confusion_mat)
    data.append(data3)
    #print(data3)

    if acc3 > best_accuracy:
        iBest = 3
        best_accuracy = acc3
    print("iBest = " + str(iBest) + "\n\n")

    print("**************4. MLP****************")
    mlp = MLPClassifier(alpha=0.05)
    mlp.fit(X_train, y_train)
    mlp_predict = mlp.predict(X_test)
    mlp_confusion_mat = confusion_matrix(y_test, mlp_predict)
    #print(mlp_confusion_mat)
    acc4 = accuracy(mlp_confusion_mat)
    rec4 = recall(mlp_confusion_mat)
    prec4 = precision(mlp_confusion_mat)
    data4 = data_serialize(4, acc4, rec4, prec4, mlp_confusion_mat)
    data.append(data4)
    #print(data4)

    if acc4 > best_accuracy:
        iBest = 4
        best_accuracy = acc4
    print("iBest = " + str(iBest) + "\n\n")

    print("**************5. Adaboost****************")
    bdt = AdaBoostClassifier()
    bdt.fit(X_train, y_train)
    bdt_predict = bdt.predict(X_test)
    bdt_confusion_mat = confusion_matrix(y_test, bdt_predict)
    #print(bdt_confusion_mat)
    acc5 = accuracy(bdt_confusion_mat)
    rec5 = recall(bdt_confusion_mat)
    prec5 = precision(bdt_confusion_mat)
    data5 = data_serialize(5, acc5, rec5, prec5, bdt_confusion_mat)
    data.append(data5)
    #print(data5)

    if acc5 > best_accuracy:
        iBest = 5
        best_accuracy = acc5
    print("iBest = " + str(iBest) + "\n\n")

    with open('a1_3.1.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(data)
    myfile.close()

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    size_list = [1000, 5000, 10000, 15000, 20000]
    accuracies = []

    if iBest == 1:
        clf = LinearSVC()
        for i in range(5):
            data_size = size_list[i]
            index = np.random.choice(X_train.shape[0], size=data_size, replace=False)
            X_train_size = X_train[index]
            y_train_size = y_train[index]
            clf.fit(X_train_size, y_train_size)
            SVC_predict = clf.predict(X_test)
            SVC_confusion_mat = confusion_matrix(y_test, SVC_predict)
            accuracies.append(accuracy(SVC_confusion_mat))
    elif iBest == 2:
        clf = SVC(gamma=2, max_iter = 1000)
        for i in range(5):
            data_size = size_list[i]
            index = np.random.choice(X_train.shape[0], size=data_size, replace=False)
            X_train_size = X_train[index]
            y_train_size = y_train[index]
            clf.fit(X_train_size, y_train_size)
            SVC_gamma_predict = clf.predict(X_test)
            SVC_gamma_confusion_mat = confusion_matrix(y_test, SVC_gamma_predict)
            accuracies.append(accuracy(SVC_gamma_confusion_mat))
    elif iBest == 3:
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        for i in range(5):
            data_size = size_list[i]
            index = np.random.choice(X_train.shape[0], size=data_size, replace=False)
            X_train_size = X_train[index]
            y_train_size = y_train[index]
            clf.fit(X_train_size, y_train_size)
            forest_predict = clf.predict(X_test)
            forest_confusion_mat = confusion_matrix(y_test, forest_predict)
            accuracies.append(accuracy(forest_confusion_mat))
    elif iBest == 4:
        clf = MLPClassifier(alpha=0.05)
        for i in range(5):
            data_size = size_list[i]
            index = np.random.choice(X_train.shape[0], size=data_size, replace=False)
            X_train_size = X_train[index]
            y_train_size = y_train[index]
            clf.fit(X_train_size, y_train_size)
            mlp_predict = clf.predict(X_test)
            mlp_confusion_mat = confusion_matrix(y_test, mlp_predict)
            accuracies.append(accuracy(mlp_confusion_mat))
    elif iBest == 5:
        clf = AdaBoostClassifier()
        for i in range(5):
            data_size = size_list[i]
            index = np.random.choice(X_train.shape[0], size=data_size, replace=False)
            X_train_size = X_train[index]
            y_train_size = y_train[index]
            clf.fit(X_train_size, y_train_size)
            bdt_predict = clf.predict(X_test)
            bdt_confusion_mat = confusion_matrix(y_test, bdt_predict)
            accuracies.append(accuracy(bdt_confusion_mat))

    #print(accuracies)

    with open('a1_3.2.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(accuracies)
    myfile.close()

    index = np.random.choice(X_train.shape[0], size=1000, replace=False)
    y_1k = y_train[index]
    X_1k = X_train[index]

    return (X_1k, y_1k)


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    k_list = [5, 10, 20, 30, 40, 50]
    data = []  # lines to be written in csv file

    # 1K training set
    pp_1k = []
    #print("1k training set: ")
    for j in range(6):
        k = k_list[j]
        #print(k)
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X_1k, y_1k)
        pp = selector.pvalues_
        feature_index = selector.get_support(indices=True)
        line = [k]
        line.extend(pp[feature_index])
        pp_1k.append(line)

    # 32K training set
    for j in range(6):
        k = k_list[j]
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X_train, y_train)
        pp = selector.pvalues_
        feature_index = selector.get_support(indices=True)
        line = [k]
        line.extend(pp[feature_index])
        data.append(line)

    if i == 1:
        clf = LinearSVC()
    elif i == 2:
        clf = SVC(gamma=2, max_iter = 1000)
    elif i == 3:
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif i == 4:
        clf = MLPClassifier(alpha=0.05)
    elif i == 5:
        clf = AdaBoostClassifier()

    accuracies = []
    # best 5 features for 1K training set
    selector = SelectKBest(f_classif, 5)
    X_train_best5 = selector.fit_transform(X_1k, y_1k)
    pp = selector.pvalues_
    feature_index = selector.get_support(indices=True)
    #print("best 5 features for 1K:")
    #print(feature_index)
    #print(pp[feature_index])
    X_test_best5 = selector.transform(X_test)
    clf.fit(X_train_best5, y_1k)
    y_predict = clf.predict(X_test_best5)
    confusion_mat = confusion_matrix(y_test, y_predict)
    accuracies.append(accuracy(confusion_mat))

    # best 5 features for 32K training set
    selector = SelectKBest(f_classif, 5)
    X_train_best5 = selector.fit_transform(X_train, y_train)
    pp = selector.pvalues_
    feature_index = selector.get_support(indices=True)
    #print("best 5 features for 32K:")
    #print(feature_index)
    #print(pp[feature_index])
    X_test_best5 = selector.transform(X_test)
    clf.fit(X_train_best5, y_train)
    y_predict = clf.predict(X_test_best5)
    confusion_mat = confusion_matrix(y_test, y_predict)
    accuracies.append(accuracy(confusion_mat))

    data.append(accuracies)

    with open('a1_3.3.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(data)
    myfile.close()
    return


def class34(filename, i):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    feats = np.load(filename)
    X = feats['arr_0'][:, 0:173]
    y = feats['arr_0'][:, 173]
    kf = KFold(n_splits=5, shuffle=True)

    data = []
    acc_list1 = []
    acc_list2 = []
    acc_list3 = []
    acc_list4 = []
    acc_list5 = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        accuracies = []

        print("**************1. linear svc****************")
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        SVC_predict = clf.predict(X_test)
        SVC_confusion_mat = confusion_matrix(y_test, SVC_predict)
        acc1 = accuracy(SVC_confusion_mat)
        accuracies.append(acc1)
        acc_list1.append(acc1)

        print("**************2. radial svc gamma = 2****************")
        clf = SVC(gamma=2, max_iter = 1000)
        clf.fit(X_train, y_train)
        SVC_gamma_predict = clf.predict(X_test)
        SVC_gamma_confusion_mat = confusion_matrix(y_test, SVC_gamma_predict)
        acc2 = accuracy(SVC_gamma_confusion_mat)
        accuracies.append(acc2)
        acc_list2.append(acc2)

        print("**************3. random forest****************")
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        clf.fit(X_train, y_train)
        forest_predict = clf.predict(X_test)
        forest_confusion_mat = confusion_matrix(y_test, forest_predict)
        acc3 = accuracy(forest_confusion_mat)
        accuracies.append(acc3)
        acc_list3.append(acc3)

        print("**************4. MLP****************")
        mlp = MLPClassifier(alpha=0.05)
        mlp.fit(X_train, y_train)
        mlp_predict = mlp.predict(X_test)
        mlp_confusion_mat = confusion_matrix(y_test, mlp_predict)
        acc4 = accuracy(mlp_confusion_mat)
        accuracies.append(acc4)
        acc_list4.append(acc4)

        print("**************5. Adaboost****************")
        bdt = AdaBoostClassifier()
        bdt.fit(X_train, y_train)
        bdt_predict = bdt.predict(X_test)
        bdt_confusion_mat = confusion_matrix(y_test, bdt_predict)
        acc5 = accuracy(bdt_confusion_mat)
        accuracies.append(acc5)
        acc_list5.append(acc5)

        print(accuracies)

        data.append(accuracies)

    p_values = []
    if i == 1:
        p_values.append(stats.ttest_rel(acc_list1, acc_list2)[1])
        p_values.append(stats.ttest_rel(acc_list1, acc_list3)[1])
        p_values.append(stats.ttest_rel(acc_list1, acc_list4)[1])
        p_values.append(stats.ttest_rel(acc_list1, acc_list5)[1])
    elif i == 2:
        p_values.append(stats.ttest_rel(acc_list2, acc_list1)[1])
        p_values.append(stats.ttest_rel(acc_list2, acc_list3)[1])
        p_values.append(stats.ttest_rel(acc_list2, acc_list4)[1])
        p_values.append(stats.ttest_rel(acc_list2, acc_list5)[1])
    elif i == 3:
        p_values.append(stats.ttest_rel(acc_list3, acc_list1)[1])
        p_values.append(stats.ttest_rel(acc_list3, acc_list2)[1])
        p_values.append(stats.ttest_rel(acc_list3, acc_list4)[1])
        p_values.append(stats.ttest_rel(acc_list3, acc_list5)[1])
    elif i == 4:
        p_values.append(stats.ttest_rel(acc_list4, acc_list1)[1])
        p_values.append(stats.ttest_rel(acc_list4, acc_list2)[1])
        p_values.append(stats.ttest_rel(acc_list4, acc_list3)[1])
        p_values.append(stats.ttest_rel(acc_list4, acc_list5)[1])
    elif i == 5:
        p_values.append(stats.ttest_rel(acc_list5, acc_list1)[1])
        p_values.append(stats.ttest_rel(acc_list5, acc_list2)[1])
        p_values.append(stats.ttest_rel(acc_list5, acc_list3)[1])
        p_values.append(stats.ttest_rel(acc_list5, acc_list4)[1])

    data.append(p_values)
    print(p_values)

    with open('a1_3.4.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(data)
    myfile.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    X_train, X_test, y_train, y_test, iBest = class31(args.input)

    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)

    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    class34(args.input, iBest)
