# coding: utf-8

from KNN import KNN
from SVM import SVM
from DT import DT
from RF import RF

if __name__ == '__main__':
    data_point = [5.1, 5.1, 5.1, 0.2]
    instance = RF("data/iris.csv", "PyCharm", max_depth=10)
    instance.read_data()
    X, Y = instance.split_data()
    print(instance.train_and_predict_cv(X, Y))
