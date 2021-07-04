from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class BaseRegressionModel(BaseEstimator):
    def train_cv(self, X, y):
        columns, dtypes = (X.columns, X.dtypes.values)
        columns_n_unique = [X[col].nunique() for col in columns]
        numeric_columns = [
            c for c, t, n in zip(columns, dtypes, columns_n_unique) if (t != 'bool') and (n > 2)
        ]

        column_transformer = ColumnTransformer(
            [
                ("standard_scale", StandardScaler(), numeric_columns)
            ],
            remainder='passthrough'
        )

        column_transformer_feature_order = numeric_columns + [
            col for col in columns if col not in numeric_columns
        ]
        final_feature_order = [columns.tolist().index(col) for col in column_transformer_feature_order]

        X_preprocessed = column_transformer.fit_transform(X)

        return cross_val_predict(
            estimator=self.model,
            X=X_preprocessed[:, final_feature_order],
            y=y,
            cv=5
        )

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
