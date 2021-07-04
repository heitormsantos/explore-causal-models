import numpy as np
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


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

        self.feature_importances_ = xt_random_search_results.best_estimator_.feature_importances_

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

        self.feature_importances_ = xt_random_search_results.best_estimator_.feature_importances_

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


class XLearnerRegressor(BaseEstimator):

    def __init__(self, covariates=[], treatment='', **kwargs):
        self._estimator_type = 'regressor'
        self.covariates = covariates
        self.treatment = treatment

        self.u0 = clone(kwargs.get('u0', XGBRegressor()))
        self.u1 = clone(kwargs.get('u1', XGBRegressor()))
        self.te_u0 = clone(kwargs.get('te_u0', XGBRegressor()))
        self.te_u1 = clone(kwargs.get('te_u1', XGBRegressor()))
        self.g = LogisticRegression(random_state=kwargs.get('random_state'))
        self.standard_scaler = StandardScaler()
        self.random_state = kwargs.get('random_state')

    def fit(self, X, y):
        w = X[self.treatment]
        self._fit_propensity_score(X)

        X_treat = X[w == 1][self.covariates].copy()
        X_control = X[w == 0][self.covariates].copy()

        y1 = y[w == 1].copy()
        y0 = y[w == 0].copy()

        self.u0 = self.u0.fit(X_control, y0)
        self.u1 = self.u1.fit(X_treat, y1)

        y1_pred = self.u1.predict(X_control)
        y0_pred = self.u0.predict(X_treat)
        te_imp_control = y1_pred - y0
        te_imp_treat = y1 - y0_pred

        self.te_u0 = self.te_u0.fit(X_control, te_imp_control)
        self.te_u1 = self.te_u1.fit(X_treat, te_imp_treat)

        self.feature_importances_ = (self.te_u1.feature_importances_ + self.te_u0.feature_importances_) / 2

    def _fit_propensity_score(self, X):
        w = X[self.treatment]
        X_scaled = self.standard_scaler.fit_transform(X[self.covariates])
        self.g = self.g.fit(X_scaled, w)

    def predict_ate(self, X):
        X_scaled = self.standard_scaler.transform(X[self.covariates])
        g_x = self.g.predict_proba(X_scaled)[:, 1]
        result = g_x * self.te_u0.predict(X[self.covariates]) + (1 - g_x) * self.te_u1.predict(X[self.covariates])
        return result

    def predict(self, X):
        # Predict y0
        y_predict_0 = self.u0.predict(X[self.covariates])

        # Predict y1
        y_predict_1 = self.u1.predict(X[self.covariates])

        return np.where(
            X[self.treatment].values == 1,
            y_predict_1,
            y_predict_0
        )