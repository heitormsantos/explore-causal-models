from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_is_fitted


class BaseRegressionModel(BaseEstimator):
    def train_cv(self, X, y):
        return cross_val_predict(self.model, X, y, cv=5)

    def get_feature_importances(self):
        check_is_fitted(self.model)
        return self.model.feature_importances_


class LinearRegressionModel(BaseRegressionModel):
    def __init__(self):
        self.model = LinearRegression()

    def get_feature_importances(self):
        check_is_fitted(self.model)
        return self.model.coef_

class ExtraTreesRegressionModel(BaseRegressionModel):
    def __init__(self):
        self.model = ExtraTreesRegressor()


class RandomForestRegressionModel(BaseRegressionModel):
    def __init__(self):
        self.model = RandomForestRegressor()
