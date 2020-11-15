# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import cross_val_score

from models_base import models_base
class SVM(models_base):
    def _init_params(self):
        params = self.params
        try:
            self.n_estimators = params["n_estimators"]
        except:
            self.n_estimators = 30
        try:
            self.max_depth = params["max_depth"]
        except:
            self.max_depth = 5
        try:
            self.cv = params["cv"]
        except:
            self.cv = 5

    def _init_model(self):
        self.model = svm.SVC()

    def train_and_predict_online(self, X, Y, data_point):
        model = self.model
        model.fit(X, Y)
        return model.predict([data_point])

    def train_and_predict_cv(self, X, Y):
        model = self.model
        scores = cross_val_score(model, X, Y, cv=5)
        return scores.mean()

    # Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
