# Copyright 2021 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The basic layout of the package is as follows:

   - The API is inspired by `scipy.stats`: `ttest_ind` and `ttest_ind_from_stats`, but does not
     match exactly
   - Each kind of test has 3 ways to call it:
      - from stats: for calling using sufficient statistics of data only
      - basic: call using the data and the control variate prediction
      - train: also train and evaluate the predictor in the routine
          - for lack of a better choice, I assume the model has a sklearn-style `fit()` and
            `predict()` API
   - Each function returns: a best estimate, a confidence interval, and p-value
      - The p-value and confidence interval should be consistent with each other
   - There are four kinds of tests here:
      - basic z-test: from the textbooks
      - cv: for control variate
      - stacked: for k-fold cross validation type setup
      - See the blog for the distinctions. In general, the train version calls -> the basic version
        calls -> from stats version.

Later:
   - An option to supply a out-of-experiment group for training the predictor
   - A streaming stats version to be more compatible with DDG: E.g. E[X], E[X^2] instead of E[X] and
     Var[X].
   - Support weighted samples, better for a bootstrap analysis
   - Make the routines need to be broadcast friendly

http://www.degeneratestate.org/posts/2018/Jan/04/reducing-the-variance-of-ab-test-using-prior-information/
"""
import warnings
from copy import deepcopy

import numpy as np
import scipy.stats as ss
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression

# TODO make 0.05?? Technically this is coverage not alpha.
# Defaults
ALPHA = 0.95
K_FOLD = 5
TRAIN_FRAC = 0.2
HEALTH_CHK_PVAL = 1e-6

MIN_SPLIT = 2  # Min data size so we can estimate mean and variance
MIN_FOLD = 2  # At least need a train and test in K-fold

# TODO offer z and t versions, t might be better for MC tests of correctness, test="z" or "t"??
# TODO type hints


# Access default numpy rng in way that is short and sphinx friendly
random = np.random.random.__self__


class PassThruPred(object):
  def __init__(self):
    pass

  def fit(self, x, y):
    n, = np.shape(y)
    assert x.shape == (n, 1)

  def predict(self, x):
    n, _ = np.shape(x)
    assert x.shape == (n, 1)
    yp = x[:, 0]
    return yp


class Cuped(object):
  def __init__(self, _ddof=1):
    self.mean_x = None
    self.mean_y = None
    self.theta = None
    self.ddof = _ddof

  def fit(self, x, y):
    n, = np.shape(y)
    assert n >= MIN_SPLIT
    assert x.shape == (n, 1)
    x = x[:, 0]

    self.mean_x = np.mean(x).item()
    self.mean_y = np.mean(y).item()
    var_x = np.var(x, ddof=self.ddof)
    if np.isclose(var_x, 0.0):
      self.theta = 0.0
    else:
      self.theta = (np.cov(x, y, ddof=self.ddof)[0, 1] / var_x).item()

  def predict(self, x):
    n, _ = np.shape(x)
    assert x.shape == (n, 1)
    yp = (x[:, 0] - self.mean_x) * self.theta + self.mean_y
    assert yp.shape == (n,)
    return yp


# ==== Validation ====


def _is_psd2(cov):
  # This works for 2x2 but not in general, unlike calling cholesky this is happy with semi-def:
  is_psd = (np.trace(cov) >= -1e-8) and (np.linalg.det(cov) >= -1e-8)
  return is_psd


def _validate_alpha(alpha):
  # Only scalars coming in, so no need to pass back an np version
  assert np.shape(alpha) == ()
  assert 0.0 < alpha
  assert alpha <= 1.0


def _validate_moments_1(mean, std, n):
  # Only scalars coming in, so no need to pass back an np version
  assert np.shape(mean) == ()
  assert np.shape(std) == ()
  assert np.shape(n) == ()
  assert std >= 0
  assert n > 0


def _validate_moments_2(mean, cov, n):
  mean = np.asarray_chkfinite(mean)
  cov = np.asarray_chkfinite(cov)

  assert np.shape(mean) == (2,)
  assert np.shape(cov) == (2, 2)
  assert np.shape(n) == ()
  assert _is_psd2(cov)
  assert n > 0
  return mean, cov, n


def _validate_data(x, y, *, paired=False, dtypes=("i", "f")):
  # We can always generalize to allow non-finite input later
  x = np.asarray_chkfinite(x)
  y = np.asarray_chkfinite(y)

  assert np.ndim(x) == 1
  assert np.ndim(y) == 1
  assert len(x) >= MIN_SPLIT
  assert len(y) >= MIN_SPLIT
  # Some routines do not work with uint or bool
  assert x.dtype.kind in dtypes
  assert y.dtype.kind in dtypes
  if paired:
    assert np.shape(x) == np.shape(y)
  return x, y


def _validate_train_data(x, x_covariates, y, y_covariates, *, k_fold=2):
  x = np.asarray_chkfinite(x)
  x_covariates = np.asarray(x_covariates)
  y = np.asarray_chkfinite(y)
  y_covariates = np.asarray(y_covariates)

  n_x, d = x_covariates.shape
  n_y, = y.shape
  assert x.shape == (n_x,)
  assert y_covariates.shape == (n_y, d)
  assert d >= 1
  assert k_fold >= MIN_FOLD
  assert n_x >= MIN_SPLIT * k_fold
  assert n_y >= MIN_SPLIT * k_fold
  return (x, x_covariates, y, y_covariates), (n_x, n_y, d)


def _validate_train_data_block(x, x_covariates, y, y_covariates):
  x = np.asarray_chkfinite(x)
  x_covariates = np.asarray(x_covariates)
  y = np.asarray_chkfinite(y)
  y_covariates = np.asarray(y_covariates)

  n_x, d = x_covariates.shape
  n_y, = y.shape
  assert x.shape == (n_x,)
  assert y_covariates.shape == (n_y, d)
  assert d >= 1
  assert n_x >= MIN_SPLIT
  assert n_y >= MIN_SPLIT
  return (x, x_covariates, y, y_covariates), (n_x, n_y, d)


# ==== Health Check Features ====


def _health_check_features(x, y, *, train_frac=TRAIN_FRAC, clf=None):
  random = np.random.RandomState(0)

  n = min(len(x), len(y))

  if len(x) < n:
    x = x[subset_idx(n, len(x), random=random), :]
  if len(y) < n:
    y = y[subset_idx(n, len(y), random=random), :]
  assert len(x) == n
  assert len(y) == n

  z = np.concatenate((x, y), axis=0)
  target = np.concatenate((np.ones(n, dtype=bool), np.zeros(n, dtype=bool)), axis=0)

  train_idx = _make_train_idx(train_frac, len(z), random=random)

  # Just hard coding default clf for now
  if clf is None:
    clf = LogisticRegression()
  clf.fit(z[train_idx, :], target[train_idx])
  pred = clf.predict(z[~train_idx, :])

  n_correct_guess = np.sum(pred == target[~train_idx])
  n_guess = len(target[~train_idx])

  pval = ss.binom_test(n_correct_guess, n_guess, 0.5)

  if pval <= HEALTH_CHK_PVAL:
    warnings.warn(f"Input features have different distribution with p = {pval}", UserWarning)


def _health_check_output(x, y):
  # KS test is not valid in the presence of dupes
  z = np.concatenate((x, y), axis=0)
  # This is faster than set operations for large z, but bloom-filter would be more mem efficient
  all_unique = len(np.unique(z)) == len(z)

  if all_unique:
    _, pval = ss.ks_2samp(x, y)
  else:
    _, _, pval = ztest(x, y)  # only check the means

  if pval <= HEALTH_CHK_PVAL:
    warnings.warn(f"Predictors have different distribution with p = {pval}", UserWarning)


# ==== Basic ====


def ztest_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, *, alpha=ALPHA):
  """Version of `ztest` that works off the sufficient statistics of the data.

  Parameters
  ----------
  mean1 : float
    The sample mean of the treatment group outcome `x`.
  std1 : float
    The sample standard deviation of the treatment group outcome.
  nobs1 : int
    The number of samples in the treatment group.
  mean2 : float
    The sample mean of the control group outcome `y`.
  std2 : float
    The sample standard deviation of the control group outcome.
  nobs2 : int
    The number of samples in the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  _validate_moments_1(mean1, std1, nobs1)
  _validate_moments_1(mean2, std2, nobs2)
  _validate_alpha(alpha)

  estimate = mean1 - mean2

  var_1 = std1 ** 2
  var_2 = std2 ** 2
  std_err = np.sqrt((var_1 / nobs1) + (var_2 / nobs2))

  # ss.norm seems pretty robust except exactly at std_err = 0 (even np.nextafter(0, 1) is ok)
  if std_err == 0.0:
    lb, ub = estimate, estimate
    pval = np.float_(estimate == 0.0)
  else:
    lb, ub = ss.norm.interval(alpha, loc=estimate, scale=std_err)
    pval = 2 * ss.norm.cdf(-np.abs(estimate), loc=0.0, scale=std_err)
  return estimate, (lb, ub), pval


def ztest(x, y, *, alpha=ALPHA, _ddof=1):
  """Standard two-sample unpaired z-test. It does not assume equal sample sizes or variances.

  Parameters
  ----------
  x : ndarray of shape (n,)
    Outcomes for the treatment group.
  y : ndarray of shape (m,)
    Outcomes for the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  x, y = _validate_data(x, y, dtypes=("b", "u", "i", "f"))
  _validate_alpha(alpha)

  R = ztest_from_stats(
    np.mean(x),
    np.std(x, ddof=_ddof),
    len(x),
    np.mean(y),
    np.std(y, ddof=_ddof),
    len(y),
    alpha=alpha,
  )
  return R


# ==== Control Variate ====


def _delta_moments(mean, cov):
  delta_mean = mean[0] - mean[1]
  delta_std = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
  return delta_mean, delta_std


def subset_idx(m, n, random=random):
  idx = np.zeros(n, dtype=bool)
  idx[:m] = True
  random.shuffle(idx)
  return idx


def _make_train_idx(frac, n, random=random):
  # There are functions in sklearn we could use to avoid needing to implement this, but we are
  # trying to avoid needing sklearn as a dep outside of the unit tests.
  assert n >= 2 * MIN_SPLIT

  n_train = int(np.ceil(np.clip(frac * n, MIN_SPLIT, n - MIN_SPLIT)).item())
  assert n_train >= MIN_SPLIT
  assert n_train <= n - MIN_SPLIT
  train_idx = subset_idx(n_train, n, random=random)
  assert np.sum(train_idx) >= MIN_SPLIT
  assert np.sum(~train_idx) >= MIN_SPLIT
  return train_idx


def ztest_cv_from_stats(mean1, cov1, nobs1, mean2, cov2, nobs2, *, alpha=ALPHA):
  """Version of `ztest_cv` that works off the sufficient statistics of the data.

  Parameters
  ----------
  mean1 : ndarray of shape (2,)
    The sample mean of the treatment group outcome and its prediction: ``[mean(x), mean(xp)]``.
  cov1 : ndarray of shape (2, 2)
    The sample covariance matrix of the treatment group outcome and its prediction:
    ``cov([x, xp])``.
  nobs1 : int
    The number of samples in the treatment group.
  mean2 : ndarray of shape (2,)
    The sample mean of the control group outcome and its prediction: ``[mean(y), mean(yp)]``.
  cov2 : ndarray of shape (2, 2)
    The sample covariance matrix of the control group outcome and its prediction: ``cov([y, yp])``.
  nobs2 : int
    The number of samples in the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  mean1, cov1, nobs1 = _validate_moments_2(mean1, cov1, nobs1)
  mean2, cov2, nobs2 = _validate_moments_2(mean2, cov2, nobs2)
  _validate_alpha(alpha)

  mean1, std1 = _delta_moments(mean1, cov1)
  mean2, std2 = _delta_moments(mean2, cov2)
  R = ztest_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, alpha=alpha)
  return R


def ztest_cv(x, xp, y, yp, *, alpha=ALPHA, health_check_output=True, _ddof=1):
  """Two-sample unpaired z-test with variance reduction using control variarates (CV). It does not
  assume equal sample sizes or variances.

  The predictions (control variates) must be derived from features that are independent of
  assignment to treatment or control. If the predictions in treatment and control have a different
  distribution then the test may be invalid.

  Parameters
  ----------
  x : ndarray of shape (n,)
    Outcomes for the treatment group.
  xp : ndarray of shape (n,)
    Predicted outcomes for the treatment group.
  y : ndarray of shape (m,)
    Outcomes for the control group.
  yp : ndarray of shape (m,)
    Predicted outcomes for the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  x, xp = _validate_data(x, xp, paired=True)
  y, yp = _validate_data(y, yp, paired=True)
  _validate_alpha(alpha)

  if health_check_output:
    _health_check_output(xp, yp)

  R = ztest(x - xp, y - yp, alpha=alpha, _ddof=_ddof)
  return R


def ztest_cv_train(
  x,
  x_covariates,
  y,
  y_covariates,
  *,
  alpha=ALPHA,
  train_frac=TRAIN_FRAC,
  health_check_input=False,
  health_check_output=True,
  clf=None,
  random=random,
  _ddof=1,
):
  """Version of `ztest_cv` that also trains the control variate predictor.

  The covariates/features must be independent of assignment to treatment or control. If the features
  in treatment and control have a different distributions then the test may be invalid.

  Parameters
  ----------
  x : ndarray of shape (n,)
    Outcomes for the treatment group.
  x_covariates : ndarray of shape (n, d)
    Covariates/features for the treatment group.
  y : ndarray of shape (m,)
    Outcomes for the control group.
  y_covariates : ndarray of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.
  train_frac : float
    The fraction of data to hold out for training the predictors. To ensure test validity, we do not
    use the same data for training the predictors and performing the test. This must be inside the
    interval range ``[0, 1]``.
  health_check_input : bool
    If ``True`` perform a health check that ensures the features have the same distribution in
    treatment and control. If not, issue a warning. It works by training a classifier to predict if
    a data point is in training or control. This can be slow for a large data set since it requires
    training a classifier.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  clf : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions.
  random : RandomState
    An optional numpy random stream can be passed in for reproducibility.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates
  )
  _validate_alpha(alpha)
  assert 0.0 <= train_frac
  assert train_frac <= 1.0

  if clf is None:
    clf = PassThruPred()

  if health_check_input:
    _health_check_features(x_covariates, y_covariates)

  # The MC calibration tests fail when we fit in-sample => we need to split.
  train_idx_x = _make_train_idx(train_frac, n_x, random=random)
  train_idx_y = _make_train_idx(train_frac, n_y, random=random)

  z_covariates = np.concatenate(
    (x_covariates[train_idx_x, :], y_covariates[train_idx_y, :]), axis=0
  )
  z = np.concatenate((x[train_idx_x], y[train_idx_y]), axis=0)
  clf.fit(z_covariates, z)

  xp = clf.predict(x_covariates[~train_idx_x, :])
  assert np.all(np.isfinite(xp))
  yp = clf.predict(y_covariates[~train_idx_y, :])
  assert np.all(np.isfinite(yp))

  R = ztest_cv(
    x[~train_idx_x],
    xp,
    y[~train_idx_y],
    yp,
    alpha=alpha,
    _ddof=_ddof,
    health_check_output=health_check_output,
  )
  return R


def ztest_in_sample_train(
  x,
  x_covariates,
  y,
  y_covariates,
  *,
  alpha=ALPHA,
  health_check_input=False,
  health_check_output=False,
  clf=None,
  random=random,
  _ddof=1,
):
  """Version of `ztest_cv` that also trains the control variate predictor.

  The covariates/features must be independent of assignment to treatment or control. If the features
  in treatment and control have a different distributions then the test may be invalid.

  Parameters
  ----------
  x : ndarray of shape (n,)
    Outcomes for the treatment group.
  x_covariates : ndarray of shape (n, d)
    Covariates/features for the treatment group.
  y : ndarray of shape (m,)
    Outcomes for the control group.
  y_covariates : ndarray of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.
  health_check_input : bool
    If ``True`` perform a health check that ensures the features have the same distribution in
    treatment and control. If not, issue a warning. It works by training a classifier to predict if
    a data point is in training or control. This can be slow for a large data set since it requires
    training a classifier.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  clf : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions.
  random : RandomState
    An optional numpy random stream can be passed in for reproducibility.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates
  )
  _validate_alpha(alpha)

  if clf is None:
    clf = PassThruPred()

  if health_check_input:
    _health_check_features(x_covariates, y_covariates)

  z_covariates = np.concatenate((x_covariates, y_covariates), axis=0)
  z = np.concatenate((x, y), axis=0)
  clf.fit(z_covariates, z)

  xp = clf.predict(x_covariates)
  assert np.all(np.isfinite(xp))
  yp = clf.predict(y_covariates)
  assert np.all(np.isfinite(yp))

  R = ztest_cv(x, xp, y, yp, alpha=alpha, _ddof=_ddof, health_check_output=health_check_output)
  return R


# ==== Implement Stacked version ====


def _pool_moments(mean, cov, nobs):
  """Warning: this routine is currently only correct for ddof=0."""
  mean = np.asarray_chkfinite(mean)
  cov = np.asarray_chkfinite(cov)
  nobs = np.asarray_chkfinite(nobs)

  n_g, d = mean.shape
  assert nobs.shape == (n_g,)
  assert cov.shape == (n_g, d, d)
  assert np.all(nobs >= MIN_SPLIT)
  assert all(_is_psd2(cc) for cc in cov)

  w = nobs / float(np.sum(nobs))
  mean_ = np.sum(w[:, None] * mean, axis=0)
  cov_ = np.sum(w[:, None, None] * cov, axis=0) + np.cov(mean.T, ddof=0, aweights=w)
  nobs_ = np.sum(nobs)
  return mean_, cov_, nobs_


def _fold_idx(n, k, random=random):
  # There are functions in sklearn we could use to avoid needing to implement this, but we are
  # trying to avoid needing sklearn as a dep outside of the unit tests.
  assert n >= k
  assert k >= 1
  idx = np.arange(n) % k
  random.shuffle(idx)
  return idx


def ztest_stacked_from_stats(mean1, cov1, nobs1, mean2, cov2, nobs2, *, alpha=ALPHA):
  """Version of `ztest_stacked` that works off the sufficient statistics of the data.

  Parameters
  ----------
  mean1 : ndarray of shape (k, 2)
    The sample mean of the treatment group outcome and its prediction: ``[mean(x), mean(xp)]``, for
    each fold in the K-fold cross validation.
  cov1 : ndarray of shape (k, 2, 2)
    The sample covariance matrix of the treatment group outcome and its prediction:
    ``cov([x, xp])``, for each fold in the K-fold cross validation.
  nobs1 : ndarray of shape (k,)
    The number of samples in the treatment group, for each fold in the K-fold cross validation.
  mean2 : ndarray of shape (k, 2)
    The sample mean of the control group outcome and its prediction: ``[mean(y), mean(yp)]``, for
    each fold in the K-fold cross validation.
  cov2 : ndarray of shape (k, 2, 2)
    The sample covariance matrix of the control group outcome and its prediction: ``cov([y, yp])``,
    for each fold in the K-fold cross validation.
  nobs2 : ndarray of shape (k,)
    The number of samples in the control group, for each fold in the K-fold cross validation.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  # _pool_moments will validate the moments
  _validate_alpha(alpha)

  mean1, cov1, nobs1 = _pool_moments(mean1, cov1, nobs1)
  mean2, cov2, nobs2 = _pool_moments(mean2, cov2, nobs2)
  R = ztest_cv_from_stats(mean1, cov1, nobs1, mean2, cov2, nobs2, alpha=alpha)
  return R


def ztest_stacked(x, xp, x_fold, y, yp, y_fold, *, alpha=ALPHA, health_check_output=True):
  """Two-sample unpaired z-test with variance reduction using the *stacked* control variarates (CV)
  method. It does not assume equal sample sizes or variances.

  The predictions (control variates) must be derived from features that are independent of
  assignment to treatment or control. If the predictions in treatment and control have a different
  distribution then the test may be invalid.

  Parameters
  ----------
  x : ndarray of shape (n,)
    Outcomes for the treatment group.
  xp : ndarray of shape (n,)
    Predicted outcomes for the treatment group derived from a cross-validation routine.
  x_fold : ndarray of shape (n,)
    The cross validation fold assignment for each data point in treatment, of `dtype` `int`.
  y : ndarray of shape (m,)
    Outcomes for the control group.
  yp : ndarray of shape (m,)
    Predicted outcomes for the control group derived from a cross-validation routine.
  y_fold : ndarray of shape (n,)
    The cross validation fold assignment for each data point in control, of `dtype` `int`.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  x, xp = _validate_data(x, xp, paired=True)
  y, yp = _validate_data(y, yp, paired=True)
  _validate_alpha(alpha)

  # Current method ignores fold index => we won't validate for now
  R = ztest_cv(x, xp, y, yp, alpha=alpha, _ddof=0, health_check_output=health_check_output)
  return R


def ztest_stacked_train(
  x,
  x_covariates,
  y,
  y_covariates,
  *,
  alpha=ALPHA,
  k_fold=K_FOLD,
  health_check_input=False,
  health_check_output=True,
  clf=None,
  random=random,
):
  """Version of `ztest_stacked` that also trains the control variate predictor.

  The covariates/features must be independent of assignment to treatment or control. If the features
  in treatment and control have a different distributions then the test may be invalid.

  Parameters
  ----------
  x : ndarray of shape (n,)
    Outcomes for the treatment group.
  x_covariates : ndarray of shape (n, d)
    Covariates/features for the treatment group.
  y : ndarray of shape (m,)
    Outcomes for the control group.
  y_covariates : ndarray of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.
  k_fold : int
    The number of folds in the cross validation: `K`.
  health_check_input : bool
    If ``True`` perform a health check that ensures the features have the same distribution in
    treatment and control. If not, issue a warning. It works by training a classifier to predict if
    a data point is in training or control. This can be slow for a large data set since it requires
    training a classifier.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  clf : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions.
  random : RandomState
    An optional numpy random stream can be passed in for reproducibility.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates, k_fold=k_fold
  )
  _validate_alpha(alpha)

  if clf is None:
    clf = PassThruPred()

  if health_check_input:
    _health_check_features(x_covariates, y_covariates)

  fold_idx_x = _fold_idx(n_x, k_fold, random=random)
  fold_idx_y = _fold_idx(n_y, k_fold, random=random)

  xp = np.zeros(n_x)
  yp = np.zeros(n_y)
  for kk in range(k_fold):
    z_covariates = np.concatenate(
      (x_covariates[fold_idx_x != kk, :], y_covariates[fold_idx_y != kk, :]), axis=0
    )
    z = np.concatenate((x[fold_idx_x != kk], y[fold_idx_y != kk]), axis=0)
    clf.fit(z_covariates, z)

    xp[fold_idx_x == kk] = clf.predict(x_covariates[fold_idx_x == kk])
    yp[fold_idx_y == kk] = clf.predict(y_covariates[fold_idx_y == kk])

  R = ztest_stacked(
    x, xp, fold_idx_x, y, yp, fold_idx_y, alpha=alpha, health_check_output=health_check_output
  )
  return R


def ztest_stacked_train_blockwise(
  x,
  x_covariates,
  y,
  y_covariates,
  *,
  alpha=ALPHA,
  k_fold=K_FOLD,
  health_check_input=False,
  health_check_output=True,
  clf=None,
  random=random,
):
  """Version of `ztest_stacked_train` that is more efficient if the fit routine scales worse than
  O(N), otherwise this will not be more efficient.

  Parameters
  ----------
  x : ndarray of shape (n,)
    Outcomes for the treatment group.
  x_covariates : ndarray of shape (n, d)
    Covariates/features for the treatment group.
  y : ndarray of shape (m,)
    Outcomes for the control group.
  y_covariates : ndarray of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.
  k_fold : int
    The number of folds in the cross validation: `K`.
  health_check_input : bool
    If ``True`` perform a health check that ensures the features have the same distribution in
    treatment and control. If not, issue a warning. It works by training a classifier to predict if
    a data point is in training or control. This can be slow for a large data set since it requires
    training a classifier.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  clf : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions.
  random : RandomState
    An optional numpy random stream can be passed in for reproducibility.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates, k_fold=k_fold
  )
  _validate_alpha(alpha)

  if clf is None:
    clf = PassThruPred()

  if health_check_input:
    _health_check_features(x_covariates, y_covariates)

  fold_idx_x = _fold_idx(n_x, k_fold, random=random)
  fold_idx_y = _fold_idx(n_y, k_fold, random=random)

  xp = np.nan + np.zeros((n_x, k_fold))
  yp = np.nan + np.zeros((n_y, k_fold))
  for kk in range(k_fold):
    z_covariates = np.concatenate(
      (x_covariates[fold_idx_x == kk, :], y_covariates[fold_idx_y == kk, :]), axis=0
    )
    z = np.concatenate((x[fold_idx_x == kk], y[fold_idx_y == kk]), axis=0)
    clf.fit(z_covariates, z)

    xp[fold_idx_x != kk, kk] = clf.predict(x_covariates[fold_idx_x != kk])
    yp[fold_idx_y != kk, kk] = clf.predict(y_covariates[fold_idx_y != kk])

  # Now average the predictions
  assert np.all(np.sum(np.isnan(xp), axis=1) == 1)
  assert np.all(np.sum(np.isnan(yp), axis=1) == 1)
  xp = np.nanmean(xp, axis=1)
  yp = np.nanmean(yp, axis=1)

  R = ztest_stacked(
    x, xp, fold_idx_x, y, yp, fold_idx_y, alpha=alpha, health_check_output=health_check_output
  )
  return R


def ztest_stacked_train_load_blockwise(data_iter, *, alpha=ALPHA, clf=None, callback=None):
  """Version of `ztest_stacked_train_blockwise` that loads the data in blocks to avoid overflowing
  memory. Using `ztest_stacked_train_blockwise` is faster if all the data fits in memory.

  Parameters
  ----------
  data_iter : iterable[callable]
    An iterable of functions, where each function returns a different cross validation fold. The
    functions should return data in the format of a tuple: ``(x, x_covariates, y, y_covariates)``.
    See the parameters of `ztest_stacked_train_blockwise` for details on the shapes of these
    variables.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.
  clf : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions.
  callback : callable
    An optional callback that gets called for each cross validation fold in the format
    ``callback(clf)``. This is sometimes useful for logging.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.
  """
  k_fold = len(data_iter)
  assert k_fold >= MIN_FOLD
  _validate_alpha(alpha)

  if clf is None:
    clf = PassThruPred()

  try:
    clf = [clone(clf) for _ in range(k_fold)]
  except TypeError:
    clf = [deepcopy(clf) for _ in range(k_fold)]

  # Train a model for each block
  for clf_, data_gen in zip(clf, data_iter):
    (x, x_covariates, y, y_covariates) = data_gen()
    (x, x_covariates, y, y_covariates), _ = _validate_train_data_block(
      x, x_covariates, y, y_covariates
    )

    z_covariates = np.concatenate((x_covariates, y_covariates), axis=0)
    z = np.concatenate((x, y), axis=0)
    clf_.fit(z_covariates, z)
    if callback is not None:
      callback(clf_)

  # Now the prediction for each block
  mean1 = np.zeros((k_fold, 2))
  cov1 = np.zeros((k_fold, 2, 2))
  nobs1 = np.zeros(k_fold)
  mean2 = np.zeros((k_fold, 2))
  cov2 = np.zeros((k_fold, 2, 2))
  nobs2 = np.zeros(k_fold)
  for kk, data_gen in enumerate(data_iter):
    (x, x_covariates, y, y_covariates) = data_gen()
    (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data_block(
      x, x_covariates, y, y_covariates
    )

    # Get predictions from each fold predictor
    xp = np.nan + np.zeros((n_x, k_fold))
    yp = np.nan + np.zeros((n_y, k_fold))
    for kk_, clf_ in enumerate(clf):
      if kk != kk_:
        xp[:, kk_] = clf_.predict(x_covariates)
        yp[:, kk_] = clf_.predict(y_covariates)

    # Now average the predictions
    assert np.all(np.sum(np.isnan(xp), axis=1) == 1)
    assert np.all(np.sum(np.isnan(yp), axis=1) == 1)
    xp = np.nanmean(xp, axis=1)
    yp = np.nanmean(yp, axis=1)

    # Now save the suff stats
    mean1[kk, 0] = np.mean(x)
    mean1[kk, 1] = np.mean(xp)
    cov1[kk, :, :] = np.cov((x, xp), ddof=0)
    nobs1[kk] = n_x
    mean2[kk, 0] = np.mean(y)
    mean2[kk, 1] = np.mean(yp)
    cov2[kk, :, :] = np.cov((y, yp), ddof=0)
    nobs2[kk] = n_y
  R = ztest_stacked_from_stats(mean1, cov1, nobs1, mean2, cov2, nobs2, alpha=alpha)
  return R


def _ztest_stacked_mlrate_train(
  x,
  x_covariates,
  y,
  y_covariates,
  *,
  alpha=ALPHA,
  k_fold=K_FOLD,
  health_check_input=False,
  health_check_output=True,
  clf=None,
  random=random,
  _ddof=1,
):
  """x is treatment here, y is control."""
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates, k_fold=k_fold
  )
  _validate_alpha(alpha)

  if health_check_input:
    _health_check_features(x_covariates, y_covariates)

  if clf is None:
    clf = PassThruPred()

  # Sample the folds
  fold_idx_x = _fold_idx(n_x, k_fold, random=random)
  fold_idx_y = _fold_idx(n_y, k_fold, random=random)

  # Setup the vars
  N = n_x + n_y
  p_hat = n_x / float(N)

  # Do the K-fold cross validation prediction
  xp = np.ones((n_x, 3))
  yp = np.zeros((n_y, 3))
  for kk in range(k_fold):
    z_covariates = np.concatenate(
      (x_covariates[fold_idx_x != kk, :], y_covariates[fold_idx_y != kk, :]), axis=0
    )
    z = np.concatenate((x[fold_idx_x != kk], y[fold_idx_y != kk]), axis=0)
    clf.fit(z_covariates, z)

    xp[fold_idx_x == kk, 1] = clf.predict(x_covariates[fold_idx_x == kk, :])
    yp[fold_idx_y == kk, 1] = clf.predict(y_covariates[fold_idx_y == kk, :])

  if health_check_output:
    _health_check_output(xp[:, 1], yp[:, 1])

  # Setup the variables for the CV regression based on the predictions
  outcome_all = np.concatenate((x, y), axis=0)
  reg_mat = np.concatenate((xp, yp), axis=0)
  # Note that reg_mat[:, 2] is just an interaction term of sorts
  reg_mat[:, 2] = reg_mat[:, 0] * (reg_mat[:, 1] - np.mean(reg_mat[:, 1]))

  # Do the regression
  reg = LinearRegression(fit_intercept=True, normalize=False)
  reg.fit(reg_mat, outcome_all)
  assert reg.coef_.shape == (3,)
  assert reg.intercept_.shape == ()
  alpha_coef = np.zeros(4)
  alpha_coef[0] = reg.intercept_
  alpha_coef[1:] = reg.coef_
  assert np.all(np.isfinite(alpha_coef))

  # Get a sample variance based on results of the regression
  fac = (alpha_coef[2] * p_hat + (alpha_coef[2] + alpha_coef[3]) * (1 - p_hat)) ** 2
  pred_var = np.var(reg_mat[:, 1], ddof=_ddof) / (p_hat * (1 - p_hat))
  base_var = np.var(y, ddof=_ddof) / (1 - p_hat) + np.var(x, ddof=_ddof) / p_hat
  sample_var = base_var - (pred_var * fac)
  assert np.isfinite(sample_var)
  if sample_var <= 0.0:
    warnings.warn(
      "MLRATE calculated a negative sample variance. The predictor is probably garbage. "
      "Falling back on a no variance reduction test.",
      UserWarning,
    )
    estimate, (lb, ub), pval = ztest(x, y, alpha=alpha, _ddof=_ddof)
    return estimate, (lb, ub), pval, True

  # Extract Gaussian parameters of sampling distribution
  estimate = alpha_coef[1]
  sample_scale = np.sqrt(sample_var / N)

  # Regular setup to get test results based on Gaussian sampling distribution
  assert sample_scale > 0.0
  lb, ub = ss.norm.interval(alpha, loc=estimate, scale=sample_scale)
  pval = 2 * ss.norm.cdf(-np.abs(estimate), loc=0.0, scale=sample_scale)
  return estimate, (lb, ub), pval, False


def ztest_stacked_mlrate_train(*args, **kwargs):
  """Very similar to `ztest_stacked_train` but uses the method of Guo et. al. to try to account for
  the correlations between cross validation folds.

  The covariates/features must be independent of assignment to treatment or control. If the features
  in treatment and control have a different distributions then the test may be invalid.

  Parameters
  ----------
  x : ndarray of shape (n,)
    Outcomes for the treatment group.
  x_covariates : ndarray of shape (n, d)
    Covariates/features for the treatment group.
  y : ndarray of shape (m,)
    Outcomes for the control group.
  y_covariates : ndarray of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.95, and must be inside the interval range
    ``(0, 1]``.
  k_fold : int
    The number of folds in the cross validation: `K`.
  health_check_input : bool
    If ``True`` perform a health check that ensures the features have the same distribution in
    treatment and control. If not, issue a warning. It works by training a classifier to predict if
    a data point is in training or control. This can be slow for a large data set since it requires
    training a classifier.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  clf : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions.
  random : RandomState
    An optional numpy random stream can be passed in for reproducibility.

  Returns
  -------
  estimate : float
    Estimate of the difference in means: ``E[x] - E[y]``.
  ci : (float, float)
    Confidence interval (with coverage `alpha`) for the estimate.
  pval : float
    The p-value under the null hypothesis H0 that ``E[x] = E[y]``.

  References
  ----------
  https://drive.google.com/file/d/153hxSPJjvVejZS8ot_W8cm6S_YGO0o88/view
  """
  estimate, (lb, ub), pval, _ = _ztest_stacked_mlrate_train(*args, **kwargs)
  return estimate, (lb, ub), pval
