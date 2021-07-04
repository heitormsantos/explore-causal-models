import graphviz
from itertools import combinations
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import SessionState
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import streamlit as st

from predictive_models import ExtraTreesRegressionModel, LinearRegressionModel, RandomForestRegressionModel
from streamlit_utils import set_page_title


# 0. HELPER FUNCTIONS AND CONSTANTS
def compute_alpha_hat(alpha_y, alpha_d, tau):
    return ((4 * tau * alpha_y) + alpha_d) / ((4 * tau) + 1)

predictive_models_options = ['Linear regression', 'ExtraTrees regressor', 'Random forest regressor']
linear_regression_model = LinearRegressionModel()
xt_regression_model = ExtraTreesRegressionModel()
rf_regression_model = RandomForestRegressionModel()

predictive_models_dict = {
    'Linear regression': linear_regression_model,
    'ExtraTrees regressor': xt_regression_model,
    'Random forest regressor': rf_regression_model
}

treatment = 'coupons'

# 1. LOAD FILES
# CSV_PATH = 'data/coupons.xls'
CSV_PATH = 'data/coupons_v2.csv'
df = pd.read_csv(CSV_PATH)

filename = CSV_PATH.split('/')[-1].rstrip('.xls')

# 2. RENDER FIRST PAGE SESSIONS
set_page_title('Predictive model')
st.title('Predicitive model')
# st.write('File used:', filename)
st.write(
    "Here's our first attempt at implementing a predictive model for the data set."
)

st.sidebar.title("Configuration")

st.sidebar.header('Choose model variables for predictive model')

pred_target = st.sidebar.selectbox(
    'Which variable is the target for the prediction model?',
    df.columns.tolist(),
    index=6
)

preselected_covariates = [cov for cov in df.columns if cov not in (pred_target, 'customer_id')]

pred_covariates = st.sidebar.multiselect(
    'Select covariates for predictive model',
    df.columns.tolist(),
    default=preselected_covariates
)

# Correlation heatmap
corr = df[pred_covariates + [pred_target]].corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

st.header('Correlation map for covariates and target variable')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=1,
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5},
    ax=ax
)
st.pyplot(fig)

session_state = SessionState.get(
    train_button_sent=False
)
session_state.train_button_sent = False

pred_algorithm_name = st.sidebar.selectbox(
    'Select algorithm for predictive model',
    predictive_models_options,
    index=0
)
pred_algorithm = predictive_models_dict[pred_algorithm_name]
train_button_sent = st.button('Train predictive model')
session_state.train_button_sent = train_button_sent

if session_state.train_button_sent:
    with st.spinner("Training ongoing"):
        y_pred = pred_algorithm.train_cv(X=df[pred_covariates], y=df[pred_target])
        y_true = df[pred_target].values

        mape_crossval = mean_absolute_percentage_error(
            y_true=y_true,
            y_pred=y_pred
        )

        r2_crossval = r2_score(
            y_true=y_true,
            y_pred=y_pred
        )

        n = len(df)  # number of observations
        k = len(pred_covariates)  # number of predictors
        adjusted_r2_crossval = 1 - ((1 - r2_crossval) * (n - 1) / (n - k - 1))

        st.write(f'Cross-validation MAPE: {round(mape_crossval, 2)}')
        st.write(f'Cross-validation R2: {round(adjusted_r2_crossval, 2)}')

        pred_algorithm.model.fit(X=df[pred_covariates], y=df[pred_target])
        feature_importances = pred_algorithm.get_feature_importances()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=pred_covariates,
            y=feature_importances,
            order=np.array(pred_covariates)[np.argsort(np.array(feature_importances))[::-1]],
            ax=ax
        )
        ax.axhline(
            np.mean(feature_importances),
            linestyle='--',
            label=f'Average feature importance ({round(np.mean(feature_importances), 2)})',
            color='black'
        )
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.header('Partial dependence plot for the test set')
        pdp = plot_partial_dependence(
            pred_algorithm.model, df[pred_covariates].sample(frac=0.05), [treatment],
            kind='both',
        )
        st.pyplot(pdp.figure_)

        top_2_features = np.array(pred_covariates)[np.argsort(np.array(abs(feature_importances)))[::-1]][:2]
        st.header('Target variable contour plot')

        # contour_2d_features_0 = st.selectbox(
        #     'Select first covariates for2D-contour plot',
        #     pred_covariates,
        #     index=pred_covariates.index(top_2_features[0])
        # )
        #
        # contour_2d_features_1 = st.selectbox(
        #     'Select second covariates for2D-contour plot',
        #     pred_covariates,
        #     index=pred_covariates.index(top_2_features[1])
        # )

        contour_2d_features_0 = top_2_features[0]
        contour_2d_features_1 = top_2_features[1]

        fig = go.Figure(data=
        go.Contour(
            z=y_pred,
            x=df[contour_2d_features_0].values,  # horizontal axis
            y=df[contour_2d_features_1].values,  # vertical axis
            colorbar={"title": 'Net value'}
        )
        )

        fig.update_layout(
            title="Target variable versus 2 most important features",
            xaxis_title=f'Variable "{contour_2d_features_0}"',
            yaxis_title=f'Variable "{contour_2d_features_1}"',
            legend_title="Target variable",
            # font=dict(
            #     family="Courier New, monospace",
            #     size=18,
            #     color="RebeccaPurple"
            # )
        )

        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True
        )

        # fig.show()
        st.plotly_chart(
            fig,
            # use_container_width=True
        )