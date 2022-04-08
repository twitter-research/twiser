# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: generic_jn
#     language: python
#     name: generic_jn
# ---

# # TWISER Demo

# +
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from twiser import twiser
# -

# ## Get the data
#
# In this demo, we use the open access data from the study [The Generalizability of Survey Experiments](https://www.ipr.northwestern.edu/documents/working-papers/2014/IPR-WP-14-19.pdf).
# The data is found on [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MUJHGR).
#
# For reproducibility, the SHA1 hashes of the full data zip and csv are:
# ```
# b1478ffac1906e5381ad3456b22f63d88c2ac6c9  dataverse_files.zip
# 465255af3a6f7fc413f53a8d4d157f01f8c6002b  Study1Data.csv
# ```
# Put the data in the same folder as the notebook.

fname = "./Study1Data.csv"

# ## Select an experiment from the study to analyze
#
# The study considered the consistency of results when surveys are done using convenience samples from various sources. In this demo, we consider applying variance reduction to analyzing the results of a single question collected via a social media survey.
#
# The study asked a question in two different ways as the intervention and treatment effect was to measure how much effect there was on the response (measured on a 1 to 7 level of agreement).
#
# Here, the response is a 1 to 7 level of agreement with:
# "Do you oppose or support the proposal to forgive student loan debt?"
# The intervention was to ask the question as either:
#
#   - **Control**: "According to the U.S. Department of Education, college student loan debt now exceeds one trillion dollars, which surpasses the total credit card debt in the United States. This has led to proposals for a student loan forgiveness program."
#   - **Treatment**: "According to the U.S. Department of Education, college student loan debt now exceeds one trillion dollars, which surpasses the total credit card debt in the United States. This has led to proposals for a student loan forgiveness program. A number of expert economic analysts suggest that a student loan forgiveness program would have serious negative effects on the economy. When individuals accept a student loan, they know they are required to pay it back. By transferring this individual responsibility and debt to the national government, the burden falls on all taxpayers and lets students avoid their financial obligations."
#
# We use three pre-experiment individual features for variance reduction and improve statistical power: sex, party affiliation, and identified political ideology. Because these feature are determinied *before* the randomized assignment to control or treatment, they can be used for variance reduction. If these features are useful for predicting an individual's response then they will increase statistical power and shrink confidence intervals.
#
# For details on the columns and codings see the master codebook in the downloaded data zip:
# ```
# Study1Codebook.docx
# ```

# +
# Columns from the csv file to use
pre_experiment_cols = ["Sex", "Party", "Ideo"]
sample_col = "Sample"
treatment_col = "LoanGroup"
response_col = "LoanSupp"

# Codes for sample groups to use from master codebook
control_code = 1
treatment_code = 2
social_media_sample_code = 6
# -

# full list of columns we will be using
cols_to_keep = pre_experiment_cols + [sample_col, treatment_col, response_col]

# ## Load the data
#
# Here we load the data.
# It takes a bit of effort to correct for some of the messiness for this kind of data and encode everything as a float.
# We need to impute some of the pre-experiment features with missing values, but as long as the distribution on pre-experiment features remains unchanged between treatment and control it does not invalidate the variance reduction analysis.

# +
# Define the non-standard missing values in csv
na_values = ["-1", "-99", ".", "?"]

# Do the load and clean
df = pd.read_csv(fname, header=0, index_col=0, na_values=na_values, low_memory=False)
df = df[cols_to_keep]  # Only keep relvant columns
df = df.apply(pd.to_numeric, errors="coerce")  # Enforce everything to be float
df[pre_experiment_cols] = df[pre_experiment_cols].fillna(df[pre_experiment_cols].mean())  # Impute missing features

# Check that was all done correctly
assert (df.dtypes == float).all()
assert not df[pre_experiment_cols].isnull().any().any()
# The current columns should all be in 1--7, this will also catch if any -1 or -99 missing values got through cleaning
assert not (df < 1).any().any()
assert not (df > 7).any().any()

# And take the data via the social media survey
df = df[df[sample_col] == social_media_sample_code]

# +
# Separate out control and treatment
is_control = df[treatment_col] == control_code
is_treatment = df[treatment_col] == treatment_code
assert not (is_control & is_treatment).any()

# Extract the arrays
x = df.loc[is_treatment, response_col].values
y = df.loc[is_control, response_col].values
x_covariates = df.loc[is_treatment, pre_experiment_cols].values
y_covariates = df.loc[is_control, pre_experiment_cols].values

# Ignore any rows where the response is missing
x_covariates = x_covariates[~np.isnan(x), :]
x = x[~np.isnan(x)]
y_covariates = y_covariates[~np.isnan(y), :]
y = y[~np.isnan(y)]
# -

# ## Setup the predictor and use TWISER
#
# We want a nonlinear predictor that won't extrapolate for poor predictions, so RF is a good choice.
# We use MSE as the criterion since the variance reduction is proportional predictive MSE.
# For reproducibility, we also fix the RF random seed.

predictor = RandomForestRegressor(criterion="squared_error", random_state=0)


def show_output(estimate, ci, pval):
    """Helper function to print the results of the statical tests in TWISER."""
    (lb, ub) = ci
    sig_mark = "*" if pval < 0.05 else ""
    print(f"estimate: {estimate:.2f} in ({lb:.2f}, {ub:.2f}), CI width of {ub - lb:.2f}, p = {pval:.4f}{sig_mark}")


# ### Basic $z$-test
#
# First, we apply the basic two-sample $z$-test included in twiser. This works basically the same as `scipy.stats.ttest_ind`.

estimate, (lb, ub), pval = twiser.ztest(x, y, alpha=0.05)
show_output(estimate, (lb, ub), pval)

# ### Variance reduction with held out data
#
# Next, we apply variance reduction where the predictor was trained on a held out 30% of the data.
# This is the easiest to show validity, but some of the added power is lost because not all data is used in the test.

estimate, (lb, ub), pval = twiser.ztest_cv_train(x, x_covariates, y, y_covariates, alpha=0.05, train_frac=0.3, predictor=predictor, random=np.random.RandomState(123))
show_output(estimate, (lb, ub), pval)

# ### Variance reduction with cross validation
#
# To be more statistically efficient we train and predict using 10-fold cross validation.
# Here, no data is wasted.
# As we can see it is a more significant result.

estimate, (lb, ub), pval = twiser.ztest_stacked_train(x, x_covariates, y, y_covariates, alpha=0.05, k_fold=10, predictor=predictor, random=np.random.RandomState(123))
show_output(estimate, (lb, ub), pval)

# ### Variance reduction in-sample
#
# In the literature it is popular to train the predictor in the same sample as the test.
# This often gives the most power.
# However, any overfitting in the predictor can also invalidate the results.

estimate, (lb, ub), pval = twiser.ztest_in_sample_train(x, x_covariates, y, y_covariates, alpha=0.05, predictor=predictor, random=np.random.RandomState(123))
show_output(estimate, (lb, ub), pval)
