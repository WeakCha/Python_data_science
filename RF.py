# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from models_base import models_base


class RF(models_base):

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
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)

    def train_and_predict_online(self, X, Y, data_point):
        self._init_params()
        self._init_model()
        model = self.model
        model.fit(X, Y)
        return model.predict([data_point])

    def train_and_predict_cv(self, X, Y):
        self._init_params()
        self._init_model()
        model = self.model
        scores = cross_val_score(model, X, Y, cv=5)
        return scores.mean()

    def print_hi(self):
        # Use a breakpoint in the code line below to debug your script.
        print(f'Hi, {self.name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
