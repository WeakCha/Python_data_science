import os
import sys
from abc import abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class models_base:
    def __init__(self, data_path="data/iris.csv", name="Pycharm", **kwargs):
        self.data_path = data_path
        self.name = name
        self.params = kwargs

    def read_data(self):
        dataset = pd.read_csv(self.data_path, encoding="utf-8")
        self.dataset = dataset
        # return dataset

    def split_data(self):
        Y = self.dataset.iloc[:, -1]
        X = self.dataset.iloc[:, 1:-1]
        return X, Y

    @abstractmethod
    def _init_params(self):
        raise NotImplementedError

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def train_and_predict_cv(self, X, Y):
        raise NotImplementedError

    @abstractmethod
    def train_and_predict_online(self, X, Y, data_point):
        raise NotImplementedError
