import graphviz
from itertools import combinations
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.inspection import plot_partial_dependence
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import umap

from causal_models import CausalExtraTreesRegressor


# 0. HELPER FUNCTIONS
def compute_alpha_hat(alpha_y, alpha_d, tau):
    return ((4 * tau * alpha_y) + alpha_d) / ((4 * tau) + 1)

# 1. LOAD FILES
CSV_PATH = 'data/low100.csv'
df = pd.read_csv(CSV_PATH)

true_ate_df = pd.read_csv(
    'data/lowDim_trueATE.csv'
)

filename = CSV_PATH.split('/')[-1].rstrip('.csv')

# 2. RENDER FIRST PAGE SESSIONS
st.title('My causal model')
st.write('File used:', filename)

st.write(
    "Here's our first attempt at implementing a causal model for an ACIC 2019 challenge data set"
)

# st.write(
#     df.head()
# )

st.header('Choose model variables')

# 2.1 CAUSAL VARIABLE SELECTION

target = st.selectbox(
    'Which variable is the target?',
    df.columns.tolist(),
    index=0
)

treatment = st.selectbox(
    'Which variable is the treatment flag?',
    df.columns.tolist(),
    index=1
)

initial_covariates = [col for col in df.columns if col not in [target, treatment]]

# 2.1.0 Exclude low variance covariates
target_coef_variation = df[target].std() / df[target].mean()
min_coef_variation = min(0.01, target_coef_variation / 10) # Minimum coefficient of variation is 10% of the target's CV

covariates_coef_variation = (df[initial_covariates].std() / df[initial_covariates].mean())

low_variance_covariates = covariates_coef_variation[
    covariates_coef_variation < min_coef_variation
].index.tolist()

proper_variance_covariates = [cov for cov in initial_covariates if cov not in low_variance_covariates]

if len(low_variance_covariates) > 0:
    st.write('These are the excluded low variance covariates:', low_variance_covariates)

# 2.1.1 Exclude highly colinear covariates
correlations = df[proper_variance_covariates + [target]].corr(method='spearman').abs()

covariates_sorted_by_target_correleation = correlations[
    ~correlations.index.isin([target])
]['Y'].abs().sort_values(ascending=False).index.tolist()

colinear_corr_threhsold = 0.8
highly_colinear_covariates = []

for v1, v2 in combinations(proper_variance_covariates, 2):
    if correlations.loc[v1, v2] > colinear_corr_threhsold:
        highly_colinear_covariates.append(
            [v1, v2][
                np.argmin(
                    [
                        covariates_sorted_by_target_correleation.index(v1), 
                        covariates_sorted_by_target_correleation.index(v2)
                    ]
                )
            ]
        )
highly_colinear_covariates = list(set(highly_colinear_covariates))

potential_covariates = [cov for cov in proper_variance_covariates if cov not in highly_colinear_covariates]

if len(highly_colinear_covariates) > 0:
    st.write('These are the excluded highly colinear covariates:', highly_colinear_covariates)

# st.write('These are the potential covariates:', potential_covariates)

# 2.1.2 Counfounding variables selection
standard_scaler = StandardScaler()
X_pot_cov_std = standard_scaler.fit_transform(df[potential_covariates])
y_treatment = df[treatment]
y_outcome = df[target]

# 2.1.2.1 Parameters for treatment prediction
treatment_prediction_model = LogisticRegression()
treatment_prediction_model.fit(X_pot_cov_std, y_treatment)
# treatment_prediction_model.coef_

# 2.1.2.2 Parameters for outcome prediction
outcome_prediction_model = LinearRegression()
outcome_prediction_model.fit(X_pot_cov_std, y_outcome)

# 2.1.2.3 Compute alpha vectors
treatment_alpha = np.abs(treatment_prediction_model.coef_[0]) / np.sum(np.abs(treatment_prediction_model.coef_[0]))
outcome_alpha = np.abs(outcome_prediction_model.coef_) / np.sum(np.abs(outcome_prediction_model.coef_))

# 2.1.2.4 Compute best estimated alpha
alpha_hat = compute_alpha_hat(
    alpha_y=outcome_alpha, 
    alpha_d=treatment_alpha, 
    tau=0.1
)

alpha_min_threshold = 1 / sqrt(len(df))

alpha_selected_covariates = np.array(potential_covariates)[
    np.where(alpha_hat > alpha_min_threshold)
].tolist()

### 

covariates = st.sidebar.multiselect(
    'Select covariates (most probable confounding covariates are automatically preselected)',
    df.columns.tolist(),
    default=alpha_selected_covariates
)

st.sidebar.header('Choose train test split ratio')
test_ratio = st.slider(
    'Choose the percentage of data to be used as test set',
    min_value=0,
    max_value=100,
    value=30,
    step=1
)

# 2.1.2.5 Check treatment flag predictability (which might indicate strong treatment assignment bias)
max_treatment_prediction_auc = 0.7
treatment_prediction_cv_model = LogisticRegression()
y_treatment_prob = cross_val_predict(
    estimator=treatment_prediction_cv_model, 
    X=X_pot_cov_std,
    y=y_treatment, 
    method='predict_proba',
    cv=5
)
roc_auc_treatment_prediction = roc_auc_score(
    y_true=y_treatment, 
    y_score=y_treatment_prob[:, 1]
)

if roc_auc_treatment_prediction > max_treatment_prediction_auc:
    st.sidebar.header('Warning: the positivity assumption might be violated!')
    st.sidebar.write(
        f'The AUC for treatment assignment prediction is {round(roc_auc_treatment_prediction, 2)}, which might indicate a violation of the positivity assumption.'
    )

# 3. SPLIT DATA SET
X_train, X_test, y_train, y_test = train_test_split(    
    df[[treatment] + covariates],
    df[target],
    test_size=test_ratio / 100,
    random_state=42
)

st.header('Set shapes')
st.write('Whole set shape:', len(df))
st.write('Train set shape:', len(X_train))
st.write('Test set shape:', len(X_test))
st.write('Number of covariates:', len(covariates))

# Create a graph object for the causal model
st.header('Causal model graph')
graph = graphviz.Digraph()

for cov in covariates:
    graph.edge(cov, treatment)
    graph.edge(cov, target)
graph.edge(treatment, target)
st.graphviz_chart(graph)

# Check covariate clusters
if st.button('Cluster samples'):
    with st.spinner('Clustering ongoing'):
        standard_scaler = StandardScaler()
        X_train_cov_std = standard_scaler.fit_transform(X_train[covariates])

        train_set_leaves_2d_embedding = X_train_cov_std
        if len(covariates) > 2:
             umap_2d_reducer = umap.UMAP(n_components=2, random_state=42)
             train_set_leaves_2d_embedding = umap_2d_reducer.fit_transform(X_train_cov_std)
        X_train[['e0', 'e1']] = train_set_leaves_2d_embedding
        X_train[target] = y_train

        st.header('Check train set covariates embedding')
        fig = px.scatter(
            X_train, 
            x="e0", 
            y="e1", 
            color=target,
        )
        st.plotly_chart(
            fig, 
        )

# 4. TRAIN MODELS

# Train ExtraTrees model with the train set
causal_model = CausalExtraTreesRegressor(covariates=covariates, treatment=treatment)
causal_model.fit(X=X_train[[treatment] + covariates], y=y_train)

model_feat_importances = pd.DataFrame(
    columns=['feature', 'importance']
)

model_feat_importances['feature'] = covariates
model_feat_importances['importance'] = causal_model.extratrees_model.feature_importances_

top_2_features = model_feat_importances.sort_values(
    'importance',
    ascending=False
)['feature'].head(2).values.tolist()

# 5. MAKE PREDICTIONS
y_predict_test = causal_model.predict(X=X_test[[treatment] + covariates])
y_predict_test_ate = causal_model.predict_ate(X=X_test[[treatment] + covariates])

# 6. DISPLAY RESULTS
mape_test = mean_absolute_percentage_error(
    y_true=y_test,
    y_pred=y_predict_test
)

r2_test = r2_score(
    y_true=y_test,
    y_pred=y_predict_test
)

n = len(X_test) # number of observations
k = len([treatment] + covariates) # number of predictors
adjusted_r2_test = 1 - ((1 - r2_test) * (n - 1) / (n - k - 1))

st.header('Model performance over test set')
st.write(
    'Test MAPE:', 
    round(
        mape_test,
        4
    )
)
st.write(
    'Adjusted test R2:', 
    round(
        adjusted_r2_test,
        4
    )
)

st.header('True and predicted ATE over test set')
st.write('True ATE:', true_ate_df.query(f'filename == "{filename}"')['trueATE'].values[0])
st.write(
    'Naïvely computed ATE:', 
    round(
        df.query(f'{treatment} == 1')['Y'].mean() - df.query(f'{treatment} == 0')['Y'].mean(),
        4
    )
)
st.write(
    'Predicted ATE:', 
    round(
        np.mean(y_predict_test_ate),
        4
    )
)

st.header('Partial dependence plot for the test set')
pdp = plot_partial_dependence(
    causal_model, X_test, treatment,
    kind='both',
    # subsample=0.6
)
st.pyplot(pdp.figure_)

st.header('Treatment effect distribution over test set')
fig, ax = plt.subplots()
ax.hist(
    y_predict_test_ate, 
    bins=20,
    # label='Predicted treatment effect distribution',
    color='midnightblue'
)
plt.axvline(
    np.mean(y_predict_test_ate), 
    linestyle='--',
    label='Predicted ATE',
    color='royalblue'
)
plt.axvline(
    true_ate_df.query(f'filename == "{filename}"')['trueATE'].values[0], 
    linestyle='--',
    label='True ATE',
    color='seagreen'
)
plt.legend()
st.pyplot(fig)

st.header('Treatment effect contour plot')
fig = go.Figure(data =
    go.Contour(
        z=y_predict_test_ate,
        x=X_test[top_2_features[0]], # horizontal axis
        y=X_test[top_2_features[1]], # vertical axis
        colorbar={"title": 'Individual treatment effect'}
    )
)

fig.update_layout(
    title=" Treatment effect versus 2 most important features on test set",
    xaxis_title=f"Variable {top_2_features[0]}",
    yaxis_title=f"Variable {top_2_features[1]}",
    legend_title="Individual treatment effect",
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
