import numpy as np
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class CausalExtraTreesRegressor(BaseEstimator):
    def __init__(self, covariates=[], treatment='', knn_params={}):
        self._estimator_type = 'regressor'
        self.covariates = covariates
        self.treatment = treatment
        
        self.knn_params = knn_params
        if not knn_params:
            self.knn_params = {
                'n_neighbors': 10, 
                'metric': 'hamming', 
            }
        
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(    
            X,
            y,
            test_size=0.5,
            random_state=42
        )

        xt_params = {
            'n_estimators': randint(10, 800),
            'max_depth': randint(3, 20),
            'max_features': randint(int(0.4 * len(self.covariates)), len(self.covariates))
        }
        xt_model = ExtraTreesRegressor(random_state=42)
        xt_random_search = RandomizedSearchCV(
            xt_model, 
            xt_params, 
            n_iter=65, 
            cv=3,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=10,
            random_state=1
        )

        xt_random_search_results = xt_random_search.fit(
            X=X_train[self.covariates], 
            y=y_train
        )

        self.extratrees_model = xt_random_search_results.best_estimator_
        self.extratrees_params_ = xt_random_search_results.best_params_

        leaves_val = self.extratrees_model.apply(X_val[self.covariates])

        # Train KNN model for the control group with the validation set
        self.knn_control = KNeighborsRegressor(
            **self.knn_params
        ).fit(
            X=leaves_val[X_val.reset_index().query(f'{self.treatment} == 0').index.tolist(), :], 
            y=y_val.loc[X_val.query(f'{self.treatment} == 0').index]
        )

        # Train KNN model for the treated with the validation set
        self.knn_treated = KNeighborsRegressor(
            **self.knn_params
        ).fit(
            X=leaves_val[X_val.reset_index().query(f'{self.treatment} == 1').index.tolist(), :], 
            y=y_val.loc[X_val.query(f'{self.treatment} == 1').index]
        )
        
    def predict(self, X):
        leaves = self.extratrees_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return np.where(
            X[self.treatment].values == 1,
            y_predict_1,
            y_predict_0
        )
    
    def predict_ate(self, X):
        leaves = self.extratrees_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return y_predict_1 - y_predict_0


class CausalExtraTreesClassifier(BaseEstimator):
    def __init__(self, covariates=[], treatment='', knn_params={}):
        self._estimator_type = 'classifier'
        self.covariates = covariates
        self.treatment = treatment
        self.classes_ = [0, 1]
        
        self.knn_params = knn_params
        if not knn_params:
            self.knn_params = {
                'n_neighbors': 10, 
                'metric': 'hamming', 
            }
        
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(    
            X,
            y,
            test_size=0.5,
            random_state=42
        )

        xt_params = {
            'n_estimators': randint(10, 800),
            'max_depth': randint(3, 20),
            'max_features': randint(int(0.4 * len(self.covariates)), len(self.covariates))
        }
        xt_model = ExtraTreesClassifier(random_state=42)
        xt_random_search = RandomizedSearchCV(
            xt_model, 
            xt_params, 
            n_iter=65, 
            cv=3,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=10,
            random_state=1
        )

        xt_random_search_results = xt_random_search.fit(
            X=X_train[self.covariates], 
            y=y_train
        )

        self.extratrees_model = xt_random_search_results.best_estimator_
        self.extratrees_params_ = xt_random_search_results.best_params_

        leaves_val = self.extratrees_model.apply(X_val[self.covariates])

        # Train KNN model for the control group with the validation set
        self.knn_control = KNeighborsClassifier(
            **self.knn_params
        ).fit(
            X=leaves_val[X_val.reset_index().query(f'{self.treatment} == 0').index.tolist(), :], 
            y=y_val.loc[X_val.query(f'{self.treatment} == 0').index]
        )

        # Train KNN model for the treated with the validation set
        self.knn_treated = KNeighborsClassifier(
            **self.knn_params
        ).fit(
            X=leaves_val[X_val.reset_index().query(f'{self.treatment} == 1').index.tolist(), :], 
            y=y_val.loc[X_val.query(f'{self.treatment} == 1').index]
        )
        
    def predict(self, X):
        leaves = self.extratrees_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return np.where(
            X[self.treatment].values == 1,
            y_predict_1,
            y_predict_0
        )

    def predict_proba(self, X):
        leaves = self.extratrees_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict_proba(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict_proba(X=leaves)

        y_prob_0 = np.where(
            X[self.treatment].values == 1,
            y_predict_1[:, 0],
            y_predict_0[:, 0]
        )

        y_prob_1 = np.where(
            X[self.treatment].values == 1,
            y_predict_1[:, 1],
            y_predict_0[:, 1]
        )

        return np.column_stack((y_prob_0, y_prob_1))
    
    def predict_ate(self, X):
        leaves = self.extratrees_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict_proba(X=leaves)[:, 1]

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict_proba(X=leaves)[:, 1]

        return y_predict_1 - y_predict_0