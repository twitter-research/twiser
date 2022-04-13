# Copyright 2021 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import scipy.stats as ss
from hypothesis import given, settings
from hypothesis.strategies import floats, integers
from hypothesis_gufunc.gufunc import gufunc_args
from sklearn.linear_model import LinearRegression
from twiser import twiser

N_TESTS_OOM = 100  # Order of magnitude on roughly how many tests in file

PVAL_CUTOFF = 0.01 / (N_TESTS_OOM * settings().max_examples)
RUNS = 1000

print(f"\nCutoff prob in MC tests P = {PVAL_CUTOFF}")

# For t-test instead of z-test we can use small_samples instead of medium_samples
small_samples = integers(min_value=1, max_value=20)
medium_samples = integers(min_value=20, max_value=50)
big_samples = integers(min_value=100, max_value=500)
easy_floats = floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0)
corrs = floats(allow_nan=False, allow_infinity=False, min_value=-0.999, max_value=0.999)
alphas = floats(min_value=0.01, max_value=0.99)
seeds = integers(min_value=0, max_value=2 ** 30)


def gen_validate(results_list, true_value, alpha, test_null=True):
  estimates, cis, pvals = zip(*results_list)

  # test unbiased
  estimates = np.array(estimates)
  n, = estimates.shape
  assert estimates.shape == (n,)
  assert np.all(np.isfinite(estimates))
  _, pval_bias = ss.ttest_1samp(estimates - true_value, 0.0)

  # test coverage
  cis = np.array(cis)
  assert cis.shape == (n, 2)
  assert np.all(np.isfinite(cis))
  lb, ub = cis[:, 0], cis[:, 1]
  assert np.all(lb <= ub)
  good_ci_count = np.sum((lb <= true_value) & (true_value <= ub))
  pval_coverage = ss.binom_test(good_ci_count, n, 1.0 - alpha)

  pvals = np.array(pvals)
  assert pvals.shape == (n,)
  assert np.all(np.isfinite(pvals))
  assert np.all(pvals >= 0)
  assert np.all(pvals <= 1)

  # test p-value is uniform
  if test_null:
    assert true_value == 0.0
    _, pval_pval = ss.kstest(pvals, "uniform", alternative="two-sided")
    pval_total = 3 * np.min([pval_bias, pval_coverage, pval_pval])
  else:
    pval_total = 2 * np.min([pval_bias, pval_coverage])

  return pval_total


@settings(deadline=None)
@given(medium_samples, medium_samples, easy_floats, easy_floats, easy_floats, alphas, seeds)
def test_ztest_null(nobs1, nobs2, std1, std2, mean, alpha, seed):
  runs = RUNS
  random = np.random.RandomState(seed)

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  results_list = []
  for _ in range(runs):
    x1 = random.randn(nobs1) * std1 + mean
    x2 = random.randn(nobs2) * std2 + mean

    R = twiser.ztest(x1, x2, alpha=alpha)
    results_list.append(R)

  pval = gen_validate(results_list, true_value=0.0, alpha=alpha)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  medium_samples, medium_samples, easy_floats, easy_floats, easy_floats, easy_floats, alphas, seeds
)
def test_ztest_alt(nobs1, nobs2, std1, std2, mean1, mean2, alpha, seed):
  runs = RUNS
  random = np.random.RandomState(seed)

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  true_value = mean1 - mean2

  results_list = []
  for _ in range(runs):
    x1 = random.randn(nobs1) * std1 + mean1
    x2 = random.randn(nobs2) * std2 + mean2

    R = twiser.ztest(x1, x2, alpha=alpha)
    results_list.append(R)

  pval = gen_validate(results_list, true_value=true_value, alpha=alpha, test_null=False)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(small_samples, small_samples, easy_floats, easy_floats, easy_floats, alphas, seeds)
def test_ztest_from_stats_null(nobs1, nobs2, std1, std2, mean, alpha, seed):
  runs = RUNS
  random = np.random.RandomState(seed)

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  results_list = []
  for _ in range(runs):
    x1 = random.randn(nobs1) * std1 + mean
    x2 = random.randn(nobs2) * std2 + mean

    R = twiser.ztest_from_stats(np.mean(x1), std1, nobs1, np.mean(x2), std2, nobs2, alpha=alpha)
    results_list.append(R)

  pval = gen_validate(results_list, true_value=0.0, alpha=alpha)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  small_samples, small_samples, easy_floats, easy_floats, easy_floats, easy_floats, alphas, seeds
)
def test_ztest_from_stats_alt(nobs1, nobs2, std1, std2, mean1, mean2, alpha, seed):
  runs = RUNS
  random = np.random.RandomState(seed)

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  true_value = mean1 - mean2

  results_list = []
  for _ in range(runs):
    x1 = random.randn(nobs1) * std1 + mean1
    x2 = random.randn(nobs2) * std2 + mean2

    R = twiser.ztest_from_stats(np.mean(x1), std1, nobs1, np.mean(x2), std2, nobs2, alpha=alpha)
    results_list.append(R)

  pval = gen_validate(results_list, true_value=true_value, alpha=alpha, test_null=False)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  medium_samples,
  medium_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  corrs,
  corrs,
  alphas,
  seeds,
)
def test_ztest_held_out_null(
  nobs1, nobs2, std1, std2, std_pred, mean_val, mean_pred, rho1, rho2, alpha, seed
):
  runs = RUNS
  random = np.random.RandomState(seed)

  std1 = np.exp(std1)
  std2 = np.exp(std2)
  std_pred = np.exp(std_pred)

  cov1 = np.zeros((2, 2))
  cov1[0, 0] = std1 ** 2
  cov1[1, 1] = std_pred ** 2
  cov1[0, 1] = rho1 * std1 * std_pred
  cov1[1, 0] = cov1[0, 1]

  cov2 = np.zeros((2, 2))
  cov2[0, 0] = std2 ** 2
  cov2[1, 1] = std_pred ** 2
  cov2[0, 1] = rho2 * std2 * std_pred
  cov2[1, 0] = cov2[0, 1]

  mean = np.array([mean_val, mean_pred])

  results_list = []
  for _ in range(runs):
    x1 = random.multivariate_normal(mean, cov1, size=nobs1)
    x2 = random.multivariate_normal(mean, cov2, size=nobs2)

    x, xp = x1[:, 0], x1[:, 1]
    y, yp = x2[:, 0], x2[:, 1]
    R = twiser.ztest_held_out(x, xp, y, yp, alpha=alpha)
    results_list.append(R)

  pval = gen_validate(results_list, true_value=0.0, alpha=alpha)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  medium_samples,
  medium_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  corrs,
  corrs,
  alphas,
  seeds,
)
def test_ztest_held_out_alt(
  nobs1, nobs2, std1, std2, std_pred, mean1, mean2, mean_pred, rho1, rho2, alpha, seed
):
  runs = RUNS
  random = np.random.RandomState(seed)

  true_value = mean1 - mean2

  std1 = np.exp(std1)
  std2 = np.exp(std2)
  std_pred = np.exp(std_pred)

  cov1 = np.zeros((2, 2))
  cov1[0, 0] = std1 ** 2
  cov1[1, 1] = std_pred ** 2
  cov1[0, 1] = rho1 * std1 * std_pred
  cov1[1, 0] = cov1[0, 1]

  cov2 = np.zeros((2, 2))
  cov2[0, 0] = std2 ** 2
  cov2[1, 1] = std_pred ** 2
  cov2[0, 1] = rho2 * std2 * std_pred
  cov2[1, 0] = cov2[0, 1]

  mean1 = np.array([mean1, mean_pred])
  mean2 = np.array([mean2, mean_pred])

  results_list = []
  for _ in range(runs):
    x1 = random.multivariate_normal(mean1, cov1, size=nobs1)
    x2 = random.multivariate_normal(mean2, cov2, size=nobs2)

    x, xp = x1[:, 0], x1[:, 1]
    y, yp = x2[:, 0], x2[:, 1]
    R = twiser.ztest_held_out(x, xp, y, yp, alpha=alpha)
    results_list.append(R)

  pval = gen_validate(results_list, true_value=true_value, alpha=alpha, test_null=False)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  small_samples,
  small_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  corrs,
  corrs,
  alphas,
  seeds,
)
def test_ztest_held_out_from_stats_null(
  nobs1, nobs2, std1, std2, std_pred, mean_val, mean_pred, rho1, rho2, alpha, seed
):
  runs = RUNS
  random = np.random.RandomState(seed)

  std1 = np.exp(std1)
  std2 = np.exp(std2)
  std_pred = np.exp(std_pred)

  cov1 = np.zeros((2, 2))
  cov1[0, 0] = std1 ** 2
  cov1[1, 1] = std_pred ** 2
  cov1[0, 1] = rho1 * std1 * std_pred
  cov1[1, 0] = cov1[0, 1]

  cov2 = np.zeros((2, 2))
  cov2[0, 0] = std2 ** 2
  cov2[1, 1] = std_pred ** 2
  cov2[0, 1] = rho2 * std2 * std_pred
  cov2[1, 0] = cov2[0, 1]

  mean = np.array([mean_val, mean_pred])

  results_list = []
  for _ in range(runs):
    x1 = random.multivariate_normal(mean, cov1, size=nobs1)
    x2 = random.multivariate_normal(mean, cov2, size=nobs2)

    R = twiser.ztest_held_out_from_stats(
      np.mean(x1, axis=0), cov1, nobs1, np.mean(x2, axis=0), cov2, nobs2, alpha=alpha
    )
    results_list.append(R)

  pval = gen_validate(results_list, true_value=0.0, alpha=alpha)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  small_samples,
  small_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  corrs,
  corrs,
  alphas,
  seeds,
)
def test_ztest_held_out_from_stats_alt(
  nobs1, nobs2, std1, std2, std_pred, mean1, mean2, mean_pred, rho1, rho2, alpha, seed
):
  runs = RUNS
  random = np.random.RandomState(seed)

  true_value = mean1 - mean2

  std1 = np.exp(std1)
  std2 = np.exp(std2)
  std_pred = np.exp(std_pred)

  cov1 = np.zeros((2, 2))
  cov1[0, 0] = std1 ** 2
  cov1[1, 1] = std_pred ** 2
  cov1[0, 1] = rho1 * std1 * std_pred
  cov1[1, 0] = cov1[0, 1]

  cov2 = np.zeros((2, 2))
  cov2[0, 0] = std2 ** 2
  cov2[1, 1] = std_pred ** 2
  cov2[0, 1] = rho2 * std2 * std_pred
  cov2[1, 0] = cov2[0, 1]

  mean1 = np.array([mean1, mean_pred])
  mean2 = np.array([mean2, mean_pred])

  results_list = []
  for _ in range(runs):
    x1 = random.multivariate_normal(mean1, cov1, size=nobs1)
    x2 = random.multivariate_normal(mean2, cov2, size=nobs2)

    R = twiser.ztest_held_out_from_stats(
      np.mean(x1, axis=0), cov1, nobs1, np.mean(x2, axis=0), cov2, nobs2, alpha=alpha
    )
    results_list.append(R)

  pval = gen_validate(results_list, true_value=true_value, alpha=alpha, test_null=False)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  medium_samples,
  medium_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  gufunc_args(
    "(d),(d,d),(d),(d)->()", dtype=np.float_, elements=easy_floats, min_side=1, max_side=5
  ),
  alphas,
  seeds,
)
def test_ztest_held_out_train_null(nobs1, nobs2, std1, std2, shift1, gu_params, alpha, seed):
  runs = RUNS
  random = np.random.RandomState(seed)

  predictor = LinearRegression()

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  mean_input, cov_input, w1, w2 = gu_params
  d, = mean_input.shape
  cov_input = np.dot(cov_input, cov_input.T)

  # Ensure null is true
  shift2 = (np.dot(w1, mean_input) + shift1) - np.dot(w2, mean_input)

  results_list = []
  for _ in range(runs):
    input1 = random.multivariate_normal(mean_input, cov_input, size=nobs1)
    assert np.all(np.isfinite(input1))
    input2 = random.multivariate_normal(mean_input, cov_input, size=nobs2)
    assert np.all(np.isfinite(input2))

    x1 = np.dot(input1, w1) + shift1 + std1 * random.randn(nobs1)
    assert x1.shape == (nobs1,)
    x2 = np.dot(input2, w2) + shift2 + std2 * random.randn(nobs2)
    assert x2.shape == (nobs2,)

    R = twiser.ztest_held_out_train(
      x1, input1, x2, input2, alpha=alpha, predictor=predictor, random=random
    )
    results_list.append(R)

  pval = gen_validate(results_list, true_value=0.0, alpha=alpha)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  medium_samples,
  medium_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  gufunc_args(
    "(d),(d,d),(d),(d)->()", dtype=np.float_, elements=easy_floats, min_side=1, max_side=5
  ),
  alphas,
  seeds,
)
def test_ztest_held_out_train_alt(nobs1, nobs2, std1, std2, shift1, shift2, gu_params, alpha, seed):
  runs = RUNS
  random = np.random.RandomState(seed)

  predictor = LinearRegression()

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  mean_input, cov_input, w1, w2 = gu_params
  d, = mean_input.shape
  cov_input = np.dot(cov_input, cov_input.T)

  # Get actual effect size
  true_value = (np.dot(w1, mean_input) + shift1) - (np.dot(w2, mean_input) + shift2)

  results_list = []
  for _ in range(runs):
    input1 = random.multivariate_normal(mean_input, cov_input, size=nobs1)
    assert np.all(np.isfinite(input1))
    input2 = random.multivariate_normal(mean_input, cov_input, size=nobs2)
    assert np.all(np.isfinite(input2))

    x1 = np.dot(input1, w1) + shift1 + std1 * random.randn(nobs1)
    assert x1.shape == (nobs1,)
    x2 = np.dot(input2, w2) + shift2 + std2 * random.randn(nobs2)
    assert x2.shape == (nobs2,)

    R = twiser.ztest_held_out_train(
      x1, input1, x2, input2, alpha=alpha, predictor=predictor, random=random
    )
    results_list.append(R)

  pval = gen_validate(results_list, true_value=true_value, alpha=alpha, test_null=False)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  medium_samples,
  medium_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  gufunc_args(
    "(d),(d,d),(d),(d)->()", dtype=np.float_, elements=easy_floats, min_side=1, max_side=5
  ),
  alphas,
  seeds,
)
def test_ztest_cross_val_train_null(nobs1, nobs2, std1, std2, shift1, gu_params, alpha, seed):
  runs = RUNS
  random = np.random.RandomState(seed)

  predictor = LinearRegression()

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  mean_input, cov_input, w1, w2 = gu_params
  d, = mean_input.shape
  cov_input = np.dot(cov_input, cov_input.T)

  # Ensure null is true
  shift2 = (np.dot(w1, mean_input) + shift1) - np.dot(w2, mean_input)

  results_list = []
  for _ in range(runs):
    input1 = random.multivariate_normal(mean_input, cov_input, size=nobs1)
    assert np.all(np.isfinite(input1))
    input2 = random.multivariate_normal(mean_input, cov_input, size=nobs2)
    assert np.all(np.isfinite(input2))

    x1 = np.dot(input1, w1) + shift1 + std1 * random.randn(nobs1)
    assert x1.shape == (nobs1,)
    x2 = np.dot(input2, w2) + shift2 + std2 * random.randn(nobs2)
    assert x2.shape == (nobs2,)

    R = twiser.ztest_cross_val_train(
      x1, input1, x2, input2, alpha=alpha, predictor=predictor, random=random
    )
    results_list.append(R)

  pval = gen_validate(results_list, true_value=0.0, alpha=alpha)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  big_samples,
  big_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  gufunc_args(
    "(d),(d,d),(d),(d)->()", dtype=np.float_, elements=easy_floats, min_side=1, max_side=5
  ),
  alphas,
  seeds,
)
def test_ztest_cross_val_train_alt(
  nobs1, nobs2, std1, std2, shift1, shift2, gu_params, alpha, seed
):
  runs = RUNS
  random = np.random.RandomState(seed)

  predictor = LinearRegression()

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  mean_input, cov_input, w1, w2 = gu_params
  d, = mean_input.shape
  cov_input = np.dot(cov_input, cov_input.T)

  # Get actual effect size
  true_value = (np.dot(w1, mean_input) + shift1) - (np.dot(w2, mean_input) + shift2)

  results_list = []
  for _ in range(runs):
    input1 = random.multivariate_normal(mean_input, cov_input, size=nobs1)
    assert np.all(np.isfinite(input1))
    input2 = random.multivariate_normal(mean_input, cov_input, size=nobs2)
    assert np.all(np.isfinite(input2))

    x1 = np.dot(input1, w1) + shift1 + std1 * random.randn(nobs1)
    assert x1.shape == (nobs1,)
    x2 = np.dot(input2, w2) + shift2 + std2 * random.randn(nobs2)
    assert x2.shape == (nobs2,)

    R = twiser.ztest_cross_val_train(
      x1, input1, x2, input2, alpha=alpha, predictor=predictor, random=random
    )
    results_list.append(R)

  pval = gen_validate(results_list, true_value=true_value, alpha=alpha, test_null=False)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  medium_samples,
  medium_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  gufunc_args(
    "(d),(d,d),(d),(d)->()", dtype=np.float_, elements=easy_floats, min_side=1, max_side=5
  ),
  alphas,
  seeds,
)
def test_ztest_cross_val_train_blockwise_null(
  nobs1, nobs2, std1, std2, shift1, gu_params, alpha, seed
):
  runs = RUNS
  random = np.random.RandomState(seed)

  predictor = LinearRegression()

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  mean_input, cov_input, w1, w2 = gu_params
  d, = mean_input.shape
  cov_input = np.dot(cov_input, cov_input.T)

  # Ensure null is true
  shift2 = (np.dot(w1, mean_input) + shift1) - np.dot(w2, mean_input)

  results_list = []
  for _ in range(runs):
    input1 = random.multivariate_normal(mean_input, cov_input, size=nobs1)
    assert np.all(np.isfinite(input1))
    input2 = random.multivariate_normal(mean_input, cov_input, size=nobs2)
    assert np.all(np.isfinite(input2))

    x1 = np.dot(input1, w1) + shift1 + std1 * random.randn(nobs1)
    assert x1.shape == (nobs1,)
    x2 = np.dot(input2, w2) + shift2 + std2 * random.randn(nobs2)
    assert x2.shape == (nobs2,)

    R = twiser.ztest_cross_val_train_blockwise(
      x1, input1, x2, input2, alpha=alpha, predictor=predictor, random=random
    )
    results_list.append(R)

  pval = gen_validate(results_list, true_value=0.0, alpha=alpha)
  print(pval)
  assert pval >= PVAL_CUTOFF


@settings(deadline=None)
@given(
  big_samples,
  big_samples,
  easy_floats,
  easy_floats,
  easy_floats,
  easy_floats,
  gufunc_args(
    "(d),(d,d),(d),(d)->()", dtype=np.float_, elements=easy_floats, min_side=1, max_side=5
  ),
  alphas,
  seeds,
)
def test_ztest_cross_val_train_blockwise_alt(
  nobs1, nobs2, std1, std2, shift1, shift2, gu_params, alpha, seed
):
  runs = RUNS
  random = np.random.RandomState(seed)

  predictor = LinearRegression()

  std1 = np.exp(std1)
  std2 = np.exp(std2)

  mean_input, cov_input, w1, w2 = gu_params
  d, = mean_input.shape
  cov_input = np.dot(cov_input, cov_input.T)

  # Get actual effect size
  true_value = (np.dot(w1, mean_input) + shift1) - (np.dot(w2, mean_input) + shift2)

  results_list = []
  for _ in range(runs):
    input1 = random.multivariate_normal(mean_input, cov_input, size=nobs1)
    assert np.all(np.isfinite(input1))
    input2 = random.multivariate_normal(mean_input, cov_input, size=nobs2)
    assert np.all(np.isfinite(input2))

    x1 = np.dot(input1, w1) + shift1 + std1 * random.randn(nobs1)
    assert x1.shape == (nobs1,)
    x2 = np.dot(input2, w2) + shift2 + std2 * random.randn(nobs2)
    assert x2.shape == (nobs2,)

    R = twiser.ztest_cross_val_train_blockwise(
      x1, input1, x2, input2, alpha=alpha, predictor=predictor, random=random
    )
    results_list.append(R)

  pval = gen_validate(results_list, true_value=true_value, alpha=alpha, test_null=False)
  print(pval)
  assert pval >= PVAL_CUTOFF
