# Copyright 2021 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The goal of this package is to make hypothesis testing using variance reduction methods as easy
as using :func:`scipy.stats.ttest_ind` and :func:`scipy.stats.ttest_ind_from_stats`. A lot of the
API is designed to match that simplicity as much as possible.

The publication in [1]_ was implented using this package. The variance reduction ideas here are
built on top of the CUPED method ideas in [2]_ and [3]_.

The package currently supports three kinds of tests:

* basic :math:`z`-test: This is the one from the intro stats textbooks.
* held out: This is a held out control variate method (train the predictor on a held out set).
* cross val: This is a :math:`k`-fold cross validation type setup when training the predictor.

The distinction between basic, held out (aka cv), and cross val (aka stacked) is discussed in [4]_.

Each method has a few different ways to call it:

* basic: Call the method using the raw data and the control variate predictions.
* from stats: Call the method using sufficient statistics of the data and predictions only.
* train: Pass in a predictor object to train and evaluate the predictor in the routine.

  * For lack of a better choice, I assume the model has a sklearn-style `fit()` and `predict()` API.

Every statistical test in this package returns the same set of variables:

* A best estimate (of the difference of means)
* A confidence interval (on the difference of means)
* A p-value under the H0 that the two means are equal

  * The p-value and confidence interval are tested to be consistent with each under inversion.

References
----------
.. [1] `R. Turner, U. Pavalanathan, S. Webb, N. Hammerla, B. Cohn, and A. Fu. Isotonic regression
   adjustment for variance reduction. In CODE@MIT, 2021
   <https://ide.mit.edu/events/2021-conference-on-digital-experimentation-mit-codemit/>`_.
.. [2] `A. Deng, Y. Xu, R. Kohavi, and T. Walker. Improving the sensitivity of online controlled
   experiments by utilizing pre-experiment data. In Proceedings of the Sixth ACM International
   Conference on Web Search and Data Mining, pages 123--132, 2013
   <https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf>`_.
.. [3] `A. Poyarkov, A. Drutsa, A. Khalyavin, G. Gusev, and P. Serdyukov. Boosted decision tree
   regression adjustment for variance reduction in online controlled experiments. In Proceedings of
   the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages
   235--244, 2016 <https://www.kdd.org/kdd2016/papers/files/adf0653-poyarkovA.pdf>`_.
.. [4] `I. Barr. Reducing the variance of A/B tests using prior information. Degenerate State, Jun
   2018
   <https://www.degeneratestate.org/posts/2018/Jan/04/reducing-the-variance-of-ab-test-using-prior-information/>`_.
"""
import warnings
from copy import deepcopy
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import scipy.stats as ss
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

# Defaults
ALPHA = 0.05
K_FOLD = 5
TRAIN_FRAC = 0.2
HEALTH_CHK_PVAL = 1e-6

MIN_SPLIT = 2  # Min data size so we can estimate mean and variance
MIN_FOLD = 2  # At least need a train and test in K-fold

# Some standard types this package uses
TestResult = Tuple[float, Tuple[float, float], float]
DataGen = Callable[[], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]
# Some placeholders that we can later make more restrictive
Model = Any
Rng = Optional[np.random.RandomState]

# Access default numpy rng in way that is short and sphinx friendly
np_random = np.random.random.__self__


class PassThruPred(object):
  def __init__(self):
    pass

  def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> None:
    n, = np.shape(y)
    assert x.shape == (n, 1)

  def predict(self, x: npt.ArrayLike) -> np.ndarray:
    n, _ = np.shape(x)
    assert x.shape == (n, 1)
    yp = x[:, 0]
    return yp


class Cuped(object):
  def __init__(self, ddof: int = 1):
    self.mean_x = None
    self.mean_y = None
    self.theta = None
    self.ddof = ddof

  def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> None:
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

  def predict(self, x: npt.ArrayLike) -> np.ndarray:
    n, _ = np.shape(x)
    assert x.shape == (n, 1)
    yp = (x[:, 0] - self.mean_x) * self.theta + self.mean_y
    assert yp.shape == (n,)
    return yp


# ==== Validation ====


def _is_psd2(cov: npt.ArrayLike) -> bool:
  # This works for 2x2 but not in general, unlike calling cholesky this is happy with semi-def:
  is_psd = (np.trace(cov) >= -1e-8) and (np.linalg.det(cov) >= -1e-8)
  return is_psd


def _validate_alpha(alpha: float) -> None:
  # Only scalars coming in, so no need to pass back an np version
  assert np.shape(alpha) == ()
  assert 0.0 <= alpha
  assert alpha < 1.0


def _validate_ddof(ddof: int) -> None:
  # Only scalars coming in, so no need to pass back an np version
  assert np.shape(ddof) == ()
  assert ddof >= 0


def _validate_moments_1(mean: float, std: float, n: int) -> None:
  # Only scalars coming in, so no need to pass back an np version
  assert np.shape(mean) == ()
  assert np.shape(std) == ()
  assert np.shape(n) == ()
  assert std >= 0
  assert n > 0


def _validate_moments_2(
  mean: npt.ArrayLike, cov: npt.ArrayLike, n: int
) -> Tuple[np.ndarray, np.ndarray, int]:
  mean = np.asarray_chkfinite(mean)
  cov = np.asarray_chkfinite(cov)

  assert np.shape(mean) == (2,)
  assert np.shape(cov) == (2, 2)
  assert np.shape(n) == ()
  assert _is_psd2(cov)
  assert n > 0
  return mean, cov, n


def _validate_data(
  x: npt.ArrayLike, y: npt.ArrayLike, *, paired: bool = False, dtypes: str = "if"
) -> Tuple[np.ndarray, np.ndarray]:
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


def _validate_train_data(
  x: npt.ArrayLike,
  x_covariates: npt.ArrayLike,
  y: npt.ArrayLike,
  y_covariates: npt.ArrayLike,
  *,
  k_fold: int = 2,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[int, int, int]]:
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


def _validate_train_data_block(
  x: npt.ArrayLike, x_covariates: npt.ArrayLike, y: npt.ArrayLike, y_covariates: npt.ArrayLike
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[int, int, int]]:
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


def _health_check_features(
  x: np.ndarray, y: np.ndarray, *, train_frac: float = TRAIN_FRAC, discriminator: Model = None
) -> None:
  random = np.random.RandomState(0)

  n = min(len(x), len(y))

  if len(x) < n:
    x = x[_subset_idx(n, len(x), random=random), :]
  if len(y) < n:
    y = y[_subset_idx(n, len(y), random=random), :]
  assert len(x) == n
  assert len(y) == n

  z = np.concatenate((x, y), axis=0)
  target = np.concatenate((np.ones(n, dtype=bool), np.zeros(n, dtype=bool)), axis=0)

  train_idx = _make_train_idx(train_frac, len(z), random=random)

  # Just hard coding default discriminator for now
  if discriminator is None:
    discriminator = LogisticRegression()
  discriminator.fit(z[train_idx, :], target[train_idx])
  pred = discriminator.predict(z[~train_idx, :])

  n_correct_guess = np.sum(pred == target[~train_idx])
  n_guess = len(target[~train_idx])

  pval = ss.binom_test(n_correct_guess, n_guess, 0.5)

  if pval <= HEALTH_CHK_PVAL:
    warnings.warn(f"Input features have different distribution with p = {pval}", UserWarning)


def _health_check_output(x: np.ndarray, y: np.ndarray) -> None:
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


def ztest_from_stats(
  mean1: float,
  std1: float,
  nobs1: int,
  mean2: float,
  std2: float,
  nobs2: int,
  *,
  alpha: float = ALPHA,
) -> TestResult:
  r"""Version of :func:`ztest` that works off the sufficient statistics of the data.

  Parameters
  ----------
  mean1 : float
    The sample mean of the treatment group outcome :math:`x`.
  std1 : float
    The sample standard deviation of the treatment group outcome.
  nobs1 : int
    The number of samples in the treatment group.
  mean2 : float
    The sample mean of the control group outcome :math:`y`.
  std2 : float
    The sample standard deviation of the control group outcome.
  nobs2 : int
    The number of samples in the control group.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
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
    lb, ub = ss.norm.interval(1.0 - alpha, loc=estimate, scale=std_err)
    pval = 2 * ss.norm.cdf(-np.abs(estimate), loc=0.0, scale=std_err)
  return estimate, (lb, ub), pval


def ztest(x: npt.ArrayLike, y: npt.ArrayLike, *, alpha: float = ALPHA, ddof: int = 1) -> TestResult:
  r"""Standard two-sample unpaired :math:`z`-test. It does not assume equal sample sizes or
  variances.

  Parameters
  ----------
  x : :class:`numpy:numpy.ndarray` of shape of shape (n,)
    Outcomes for the treatment group.
  y : :class:`numpy:numpy.ndarray` of shape (m,)
    Outcomes for the control group.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.
  ddof : int
    The "Delta Degrees of Freedom" argument for computing sample variances.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  x, y = _validate_data(x, y, dtypes="buif")
  _validate_alpha(alpha)
  _validate_ddof(ddof)

  R = ztest_from_stats(
    np.mean(x), np.std(x, ddof=ddof), len(x), np.mean(y), np.std(y, ddof=ddof), len(y), alpha=alpha
  )
  return R


# ==== Control Variate ====


def _delta_moments(mean: np.ndarray, cov: np.ndarray) -> Tuple[float, float]:
  delta_mean = mean[0] - mean[1]
  delta_std = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
  return delta_mean, delta_std


def _subset_idx(m: int, n: int, random: Rng = np_random) -> np.ndarray:
  idx = np.zeros(n, dtype=bool)
  idx[:m] = True
  random.shuffle(idx)
  return idx


def _make_train_idx(frac: float, n: int, random: Rng = np_random) -> np.ndarray:
  # There are functions in sklearn we could use to avoid needing to implement this, but we are
  # trying to avoid needing sklearn as a dep outside of the unit tests.
  assert n >= 2 * MIN_SPLIT

  n_train = int(np.ceil(np.clip(frac * n, MIN_SPLIT, n - MIN_SPLIT)).item())
  assert n_train >= MIN_SPLIT
  assert n_train <= n - MIN_SPLIT
  train_idx = _subset_idx(n_train, n, random=random)
  assert np.sum(train_idx) >= MIN_SPLIT
  assert np.sum(~train_idx) >= MIN_SPLIT
  return train_idx


def ztest_held_out_from_stats(
  mean1: npt.ArrayLike,
  cov1: npt.ArrayLike,
  nobs1: int,
  mean2: npt.ArrayLike,
  cov2: npt.ArrayLike,
  nobs2: int,
  *,
  alpha: float = ALPHA,
) -> TestResult:
  r"""Version of :func:`ztest_held_out` that works off the sufficient statistics of the data.

  Parameters
  ----------
  mean1 : :class:`numpy:numpy.ndarray` of shape (2,)
    The sample mean of the treatment group outcome and its prediction: ``[mean(x), mean(xp)]``.
  cov1 : :class:`numpy:numpy.ndarray` of shape (2, 2)
    The sample covariance matrix of the treatment group outcome and its prediction:
    ``cov([x, xp])``.
  nobs1 : int
    The number of samples in the treatment group.
  mean2 : :class:`numpy:numpy.ndarray` of shape (2,)
    The sample mean of the control group outcome and its prediction: ``[mean(y), mean(yp)]``.
  cov2 : :class:`numpy:numpy.ndarray` of shape (2, 2)
    The sample covariance matrix of the control group outcome and its prediction: ``cov([y, yp])``.
  nobs2 : int
    The number of samples in the control group.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  mean1, cov1, nobs1 = _validate_moments_2(mean1, cov1, nobs1)
  mean2, cov2, nobs2 = _validate_moments_2(mean2, cov2, nobs2)
  _validate_alpha(alpha)

  mean1, std1 = _delta_moments(mean1, cov1)
  mean2, std2 = _delta_moments(mean2, cov2)
  R = ztest_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, alpha=alpha)
  return R


def ztest_held_out(
  x: npt.ArrayLike,
  xp: npt.ArrayLike,
  y: npt.ArrayLike,
  yp: npt.ArrayLike,
  *,
  alpha: float = ALPHA,
  health_check_output: bool = True,
  ddof: int = 1,
) -> TestResult:
  r"""Two-sample unpaired :math:`z`-test with variance reduction using control variarates. It
  does not assume equal sample sizes or variances.

  The predictions (control variates) must be derived from features that are independent of
  assignment to treatment or control. If the predictions in treatment and control have a different
  distribution then the test may be invalid.

  Parameters
  ----------
  x : :class:`numpy:numpy.ndarray` of shape (n,)
    Outcomes for the treatment group.
  xp : :class:`numpy:numpy.ndarray` of shape (n,)
    Predicted outcomes for the treatment group.
  y : :class:`numpy:numpy.ndarray` of shape (m,)
    Outcomes for the control group.
  yp : :class:`numpy:numpy.ndarray` of shape (m,)
    Predicted outcomes for the control group.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  ddof : int
    The "Delta Degrees of Freedom" argument for computing sample variances.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  x, xp = _validate_data(x, xp, paired=True)
  y, yp = _validate_data(y, yp, paired=True)
  _validate_alpha(alpha)
  _validate_ddof(ddof)

  if health_check_output:
    _health_check_output(xp, yp)

  R = ztest(x - xp, y - yp, alpha=alpha, ddof=ddof)
  return R


def ztest_held_out_train(
  x: npt.ArrayLike,
  x_covariates: npt.ArrayLike,
  y: npt.ArrayLike,
  y_covariates: npt.ArrayLike,
  *,
  alpha: float = ALPHA,
  train_frac: float = TRAIN_FRAC,
  health_check_input: bool = False,
  health_check_output: bool = True,
  predictor: Model = None,
  random: Rng = None,
  ddof: int = 1,
) -> TestResult:
  r"""Version of :func:`ztest_held_out` that also trains the control variate predictor.

  The covariates/features must be independent of assignment to treatment or control. If the features
  in treatment and control have a different distribution then the test may be invalid.

  Parameters
  ----------
  x : :class:`numpy:numpy.ndarray` of shape (n,)
    Outcomes for the treatment group.
  x_covariates : :class:`numpy:numpy.ndarray` of shape (n, d)
    Covariates/features for the treatment group.
  y : :class:`numpy:numpy.ndarray` of shape (m,)
    Outcomes for the control group.
  y_covariates : :class:`numpy:numpy.ndarray` of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.
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
  predictor : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions. The object does not need
    to be fit yet. It will be fit in this method.
  random : :class:`numpy:numpy.random.RandomState`
    An optional numpy random stream can be passed in for reproducibility.
  ddof : int
    The "Delta Degrees of Freedom" argument for computing sample variances.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates
  )
  _validate_alpha(alpha)
  assert 0.0 <= train_frac
  assert train_frac <= 1.0
  _validate_ddof(ddof)

  if predictor is None:
    predictor = PassThruPred()

  if random is None:
    random = np_random

  if health_check_input:
    _health_check_features(x_covariates, y_covariates)

  # The MC calibration tests fail when we fit in-sample => we need to split.
  train_idx_x = _make_train_idx(train_frac, n_x, random=random)
  train_idx_y = _make_train_idx(train_frac, n_y, random=random)

  z_covariates = np.concatenate(
    (x_covariates[train_idx_x, :], y_covariates[train_idx_y, :]), axis=0
  )
  z = np.concatenate((x[train_idx_x], y[train_idx_y]), axis=0)
  predictor.fit(z_covariates, z)

  xp = predictor.predict(x_covariates[~train_idx_x, :])
  assert np.all(np.isfinite(xp))
  yp = predictor.predict(y_covariates[~train_idx_y, :])
  assert np.all(np.isfinite(yp))

  R = ztest_held_out(
    x[~train_idx_x],
    xp,
    y[~train_idx_y],
    yp,
    alpha=alpha,
    ddof=ddof,
    health_check_output=health_check_output,
  )
  return R


def ztest_in_sample_train(
  x: npt.ArrayLike,
  x_covariates: npt.ArrayLike,
  y: npt.ArrayLike,
  y_covariates: npt.ArrayLike,
  *,
  alpha: float = ALPHA,
  health_check_input: bool = False,
  health_check_output: bool = False,
  predictor: Model = None,
  random: Rng = None,
  ddof: int = 1,
) -> TestResult:
  r"""Version of :func:`ztest_held_out` that also trains the control variate predictor.

  The covariates/features must be independent of assignment to treatment or control. If the features
  in treatment and control have a different distribution then the test may be invalid.

  Parameters
  ----------
  x : :class:`numpy:numpy.ndarray` of shape (n,)
    Outcomes for the treatment group.
  x_covariates : :class:`numpy:numpy.ndarray` of shape (n, d)
    Covariates/features for the treatment group.
  y : :class:`numpy:numpy.ndarray` of shape (m,)
    Outcomes for the control group.
  y_covariates : :class:`numpy:numpy.ndarray` of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.
  health_check_input : bool
    If ``True`` perform a health check that ensures the features have the same distribution in
    treatment and control. If not, issue a warning. It works by training a classifier to predict if
    a data point is in training or control. This can be slow for a large data set since it requires
    training a classifier.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  predictor : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions. The object does not need
    to be fit yet. It will be fit in this method.
  random : :class:`numpy:numpy.random.RandomState`
    An optional numpy random stream can be passed in for reproducibility.
  ddof : int
    The "Delta Degrees of Freedom" argument for computing sample variances.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates
  )
  _validate_alpha(alpha)
  _validate_ddof(ddof)

  if predictor is None:
    predictor = PassThruPred()

  if random is None:
    random = np_random

  if health_check_input:
    _health_check_features(x_covariates, y_covariates)

  z_covariates = np.concatenate((x_covariates, y_covariates), axis=0)
  z = np.concatenate((x, y), axis=0)
  predictor.fit(z_covariates, z)

  xp = predictor.predict(x_covariates)
  assert np.all(np.isfinite(xp))
  yp = predictor.predict(y_covariates)
  assert np.all(np.isfinite(yp))

  R = ztest_held_out(x, xp, y, yp, alpha=alpha, ddof=ddof, health_check_output=health_check_output)
  return R


# ==== Implement cross val version ====


def _pool_moments(
  mean: npt.ArrayLike, cov: npt.ArrayLike, nobs: npt.ArrayLike
) -> Tuple[np.ndarray, np.ndarray, int]:
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


def _fold_idx(n: int, k: int, random: Rng = np_random) -> np.ndarray:
  # There are functions in sklearn we could use to avoid needing to implement this, but we are
  # trying to avoid needing sklearn as a dep outside of the unit tests.
  assert n >= k
  assert k >= 1
  idx = np.arange(n) % k
  random.shuffle(idx)
  return idx


def ztest_cross_val_from_stats(
  mean1: npt.ArrayLike,
  cov1: npt.ArrayLike,
  nobs1: npt.ArrayLike,
  mean2: npt.ArrayLike,
  cov2: npt.ArrayLike,
  nobs2: npt.ArrayLike,
  *,
  alpha: float = ALPHA,
) -> TestResult:
  r"""Version of :func:`ztest_cross_val` that works off the sufficient statistics of the data.

  Parameters
  ----------
  mean1 : :class:`numpy:numpy.ndarray` of shape (k, 2)
    The sample mean of the treatment group outcome and its prediction: ``[mean(x), mean(xp)]``, for
    each fold in the :math:`k`-fold cross validation.
  cov1 : :class:`numpy:numpy.ndarray` of shape (k, 2, 2)
    The sample covariance matrix of the treatment group outcome and its prediction:
    ``cov([x, xp])``, for each fold in the :math:`k`-fold cross validation.
  nobs1 : :class:`numpy:numpy.ndarray` of shape (k,)
    The number of samples in the treatment group, for each fold in the :math:`k`-fold cross
    validation.
  mean2 : :class:`numpy:numpy.ndarray` of shape (k, 2)
    The sample mean of the control group outcome and its prediction: ``[mean(y), mean(yp)]``, for
    each fold in the :math:`k`-fold cross validation.
  cov2 : :class:`numpy:numpy.ndarray` of shape (k, 2, 2)
    The sample covariance matrix of the control group outcome and its prediction: ``cov([y, yp])``,
    for each fold in the :math:`k`-fold cross validation.
  nobs2 : :class:`numpy:numpy.ndarray` of shape (k,)
    The number of samples in the control group, for each fold in the :math:`k`-fold cross
    validation.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  # _pool_moments will validate the moments
  _validate_alpha(alpha)

  mean1, cov1, nobs1 = _pool_moments(mean1, cov1, nobs1)
  mean2, cov2, nobs2 = _pool_moments(mean2, cov2, nobs2)
  R = ztest_held_out_from_stats(mean1, cov1, nobs1, mean2, cov2, nobs2, alpha=alpha)
  return R


def ztest_cross_val(
  x: npt.ArrayLike,
  xp: npt.ArrayLike,
  x_fold: npt.ArrayLike,
  y: npt.ArrayLike,
  yp: npt.ArrayLike,
  y_fold: npt.ArrayLike,
  *,
  alpha: float = ALPHA,
  health_check_output: bool = True,
) -> TestResult:
  r"""Two-sample unpaired :math:`z`-test with variance reduction using the cross validated control
  variarates method. It does not assume equal sample sizes or variances.

  The predictions (control variates) must be derived from features that are independent of
  assignment to treatment or control. If the predictions in treatment and control have a different
  distribution then the test may be invalid.

  Parameters
  ----------
  x : :class:`numpy:numpy.ndarray` of shape (n,)
    Outcomes for the treatment group.
  xp : :class:`numpy:numpy.ndarray` of shape (n,)
    Predicted outcomes for the treatment group derived from a cross-validation routine.
  x_fold : :class:`numpy:numpy.ndarray` of shape (n,)
    The cross validation fold assignment for each data point in treatment (of `dtype` `int`).
  y : :class:`numpy:numpy.ndarray` of shape (m,)
    Outcomes for the control group.
  yp : :class:`numpy:numpy.ndarray` of shape (m,)
    Predicted outcomes for the control group derived from a cross-validation routine.
  y_fold : :class:`numpy:numpy.ndarray` of shape (n,)
    The cross validation fold assignment for each data point in control (of `dtype` `int`).
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  x, xp = _validate_data(x, xp, paired=True)
  y, yp = _validate_data(y, yp, paired=True)
  _validate_alpha(alpha)

  # Current method ignores fold index => we won't validate for now
  R = ztest_held_out(x, xp, y, yp, alpha=alpha, ddof=0, health_check_output=health_check_output)
  return R


def ztest_cross_val_train(
  x: npt.ArrayLike,
  x_covariates: npt.ArrayLike,
  y: npt.ArrayLike,
  y_covariates: npt.ArrayLike,
  *,
  alpha: float = ALPHA,
  k_fold: int = K_FOLD,
  health_check_input: bool = False,
  health_check_output: bool = True,
  predictor: Model = None,
  random: Rng = None,
) -> TestResult:
  r"""Version of :func:`ztest_cross_val` that also trains the control variate predictor.

  The covariates/features must be independent of assignment to treatment or control. If the features
  in treatment and control have a different distribution then the test may be invalid.

  Parameters
  ----------
  x : :class:`numpy:numpy.ndarray` of shape (n,)
    Outcomes for the treatment group.
  x_covariates : :class:`numpy:numpy.ndarray` of shape (n, d)
    Covariates/features for the treatment group.
  y : :class:`numpy:numpy.ndarray` of shape (m,)
    Outcomes for the control group.
  y_covariates : :class:`numpy:numpy.ndarray` of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.
  k_fold : int
    The number of folds in the cross validation: :math:`k`.
  health_check_input : bool
    If ``True`` perform a health check that ensures the features have the same distribution in
    treatment and control. If not, issue a warning. It works by training a classifier to predict if
    a data point is in training or control. This can be slow for a large data set since it requires
    training a classifier.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  predictor : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions. The object does not need
    to be fit yet. It will be fit in this method.
  random : :class:`numpy:numpy.random.RandomState`
    An optional numpy random stream can be passed in for reproducibility.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates, k_fold=k_fold
  )
  _validate_alpha(alpha)

  if predictor is None:
    predictor = PassThruPred()

  if random is None:
    random = np_random

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
    predictor.fit(z_covariates, z)

    xp[fold_idx_x == kk] = predictor.predict(x_covariates[fold_idx_x == kk])
    yp[fold_idx_y == kk] = predictor.predict(y_covariates[fold_idx_y == kk])

  R = ztest_cross_val(
    x, xp, fold_idx_x, y, yp, fold_idx_y, alpha=alpha, health_check_output=health_check_output
  )
  return R


def ztest_cross_val_train_blockwise(
  x: npt.ArrayLike,
  x_covariates: npt.ArrayLike,
  y: npt.ArrayLike,
  y_covariates: npt.ArrayLike,
  *,
  alpha: float = ALPHA,
  k_fold: int = K_FOLD,
  health_check_input: bool = False,
  health_check_output: bool = True,
  predictor: Model = None,
  random: Rng = None,
) -> TestResult:
  r"""Version of :func:`ztest_cross_val_train` that is more efficient if the fit routine scales worse
  than O(N), otherwise this will not be more efficient.

  Parameters
  ----------
  x : :class:`numpy:numpy.ndarray` of shape (n,)
    Outcomes for the treatment group.
  x_covariates : :class:`numpy:numpy.ndarray` of shape (n, d)
    Covariates/features for the treatment group.
  y : :class:`numpy:numpy.ndarray` of shape (m,)
    Outcomes for the control group.
  y_covariates : :class:`numpy:numpy.ndarray` of shape (m, d)
    Covariates/features for the control group.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.
  k_fold : int
    The number of folds in the cross validation: :math:`k`.
  health_check_input : bool
    If ``True`` perform a health check that ensures the features have the same distribution in
    treatment and control. If not, issue a warning. It works by training a classifier to predict if
    a data point is in training or control. This can be slow for a large data set since it requires
    training a classifier.
  health_check_output : bool
    If ``True`` perform a health check that ensures the predictions have the same distribution in
    treatment and control. If not, issue a warning.
  predictor : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions. The object does not need
    to be fit yet. It will be fit in this method.
  random : :class:`numpy:numpy.random.RandomState`
    An optional numpy random stream can be passed in for reproducibility.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  (x, x_covariates, y, y_covariates), (n_x, n_y, _) = _validate_train_data(
    x, x_covariates, y, y_covariates, k_fold=k_fold
  )
  _validate_alpha(alpha)

  if predictor is None:
    predictor = PassThruPred()

  if random is None:
    random = np_random

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
    predictor.fit(z_covariates, z)

    xp[fold_idx_x != kk, kk] = predictor.predict(x_covariates[fold_idx_x != kk])
    yp[fold_idx_y != kk, kk] = predictor.predict(y_covariates[fold_idx_y != kk])

  # Now average the predictions
  assert np.all(np.sum(np.isnan(xp), axis=1) == 1)
  assert np.all(np.sum(np.isnan(yp), axis=1) == 1)
  xp = np.nanmean(xp, axis=1)
  yp = np.nanmean(yp, axis=1)

  R = ztest_cross_val(
    x, xp, fold_idx_x, y, yp, fold_idx_y, alpha=alpha, health_check_output=health_check_output
  )
  return R


def ztest_cross_val_train_load_blockwise(
  data_iter: Sequence[DataGen],
  *,
  alpha: float = ALPHA,
  predictor: Model = None,
  callback: Optional[Callable[[Model], None]] = None,
) -> TestResult:
  r"""Version of :func:`ztest_cross_val_train_blockwise` that loads the data in blocks to avoid
  overflowing memory. Using :func:`ztest_cross_val_train_blockwise` is faster if all the data fits
  in memory.

  Parameters
  ----------
  data_iter : Sequence[Callable[[], Tuple[ndarray, ndarray, ndarray, ndarray]]]
    An iterable of functions, where each function returns a different cross validation fold. The
    functions should return data in the format of a tuple: ``(x, x_covariates, y, y_covariates)``.
    See the parameters of :func:`ztest_cross_val_train_blockwise` for details on the shapes of these
    variables.
  alpha : float
    Required confidence level, typically this should be 0.05, and must be inside the interval range
    :math:`[0, 1)`.
  predictor : sklearn-like regression object
    An object that has a `fit` and `predict` routine to make predictions. The object does not need
    to be fit yet. It will be fit in this method.
  callback :
    An optional callback that gets called for each cross validation fold in the format
    ``callback(predictor)``. This is sometimes useful for logging.

  Returns
  -------
  estimate :
    Estimate of the difference in means: :math:`\mathbb{E}[x] - \mathbb{E}[y]`.
  ci :
    Confidence interval (with coverage :math:`1 - \alpha`) for the estimate.
  pval :
    The p-value under the null hypothesis H0 that :math:`\mathbb{E}[x] = \mathbb{E}[y]`.
  """
  # The dtype for best sphinx generation on data_iter is:
  # Sequence[Callable[[], Tuple[:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`,
  #   :class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`]]
  # but that messes up the max line length.
  k_fold = len(data_iter)
  assert k_fold >= MIN_FOLD
  _validate_alpha(alpha)

  if predictor is None:
    predictor = PassThruPred()

  try:
    predictor = [clone(predictor) for _ in range(k_fold)]
  except TypeError:
    predictor = [deepcopy(predictor) for _ in range(k_fold)]

  # Train a model for each block
  for predictor_, data_gen in zip(predictor, data_iter):
    (x, x_covariates, y, y_covariates) = data_gen()
    (x, x_covariates, y, y_covariates), _ = _validate_train_data_block(
      x, x_covariates, y, y_covariates
    )

    z_covariates = np.concatenate((x_covariates, y_covariates), axis=0)
    z = np.concatenate((x, y), axis=0)
    predictor_.fit(z_covariates, z)
    if callback is not None:
      callback(predictor_)

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
    for kk_, predictor_ in enumerate(predictor):
      if kk != kk_:
        xp[:, kk_] = predictor_.predict(x_covariates)
        yp[:, kk_] = predictor_.predict(y_covariates)

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
  R = ztest_cross_val_from_stats(mean1, cov1, nobs1, mean2, cov2, nobs2, alpha=alpha)
  return R
