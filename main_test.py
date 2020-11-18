# coding: utf-8

from KNN import KNN
from SVM import SVM
from DT import DT
from RF import RF
from LGB import LGB

if __name__ == '__main__':
    data_point = [5.1, 5.1, 5.1, 0.2]
    instance = LGB("data/Iris.csv", "PyCharm", n_estimators=30, max_depth=10)
    instance.read_data()
    X, Y = instance.split_data()
    print(instance.train_and_predict_online(X, Y, data_point))

