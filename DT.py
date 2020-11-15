# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

class DT:
    def __init__(self, data_path="data/iris.csv", name="Pycharm"):
        self.data_path = data_path
        self.name = name

    def read_data(self):
        dataset = pd.read_csv(self.data_path, encoding="utf-8")
        self.dataset = dataset
        # return dataset

    def split_data(self):
        Y = self.dataset['Species']
        X = self.dataset.iloc[:, 1:-1]
        return X, Y

    def train_and_predict(self, X, Y, data_point):
        model = DecisionTreeClassifier()
        model.fit(X, Y)
        scores = cross_val_score(model, X, Y, cv=5)
        return model.predict([data_point]), scores.mean()

    # Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
