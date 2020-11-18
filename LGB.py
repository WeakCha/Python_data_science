# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import heapq

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from models_base import models_base
import lightgbm as lgb

class LGB(models_base):
    def _init_params(self):
        pass

    def _init_model(self):
        pass

    def train_and_predict_online(self, X, Y, data_point):
        # model = self.model
        le = LabelEncoder()
        le.fit(Y)
        Y_new = le.transform(Y)
        data = lgb.Dataset(X, Y_new)
        bst = lgb.train(self.params, data, 100, valid_sets=data)
        return bst.predict([data_point])

    def train_and_predict_cv(self, X, Y):
        model = self.model
        scores = cross_val_score(model, X, Y, cv=5)
        return scores.mean()

    # Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
