# Copyright 2021 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from functools import partial

import numpy as np
from hypothesis import assume, given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import booleans, floats, integers, lists
from hypothesis_gufunc.gufunc import gufunc_args
from sklearn.linear_model import LinearRegression, Ridge
from wiser import wiser

easy_floats = floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0)

data_vectors = arrays(
  dtype=np.float_,
  shape=array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=20),
  elements=easy_floats,
)

data_vectors_int = (
  arrays(
    dtype=np.int_,
    shape=array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=20),
    elements=integers(-100, 100),
  )
  | arrays(
    dtype=np.uint,
    shape=array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=20),
    elements=integers(0, 100),
  )
  | arrays(
    dtype=np.bool_,
    shape=array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=20),
    elements=booleans(),
  )
)

data_vector_pairs_2 = gufunc_args(
  "(n),(n)->()", dtype=np.float_, elements=easy_floats, min_side=2, max_side=20
)

data_vector_pairs_2_int = gufunc_args(
  "(n),(n)->()", dtype=np.int_, elements=integers(-100, 100), min_side=2, max_side=20
)

data_vector_pairs_4 = gufunc_args(
  "(n),(n)->()", dtype=np.float_, elements=easy_floats, min_side=4, max_side=20
)

alphas = floats(min_value=1e-8, max_value=1.0)
seeds = integers(min_value=0, max_value=2 ** 30)
ddofs = integers(0, 1)
folds = integers(2, 10)
train_fracs = floats(min_value=0.0, max_value=1.0)


def general_hyp_tester(test_f, alpha, *args, **kwargs):
  estimate, (lb, ub), pval = test_f(*args, **kwargs, alpha=alpha)

  assert lb <= ub
  assert lb <= estimate
  assert estimate <= ub
  assert 0.0 <= pval and pval <= 1.0

  # If the p-value is too extreme, the inversion test will fail due to numerics and not any other
  # implementation error.
  if pval <= 1e-8 or pval >= 1 - 1e-8:
    return

  # Now test "inversion"
  estimate_, (lb_, ub_), pval_ = test_f(*args, **kwargs, alpha=1.0 - pval)

  # These should be invariant to alpha
  assert estimate == estimate_
  assert pval == pval_

  if estimate <= 0.0:
    assert np.isclose(ub_, 0.0)
  else:
    assert np.isclose(lb_, 0.0)


def general_float_test(test_f, *args, **kwargs):
  estimate, (lb, ub), pval = test_f(*args, **kwargs)

  args = [np.asarray(aa, dtype=np.float_) for aa in args]
  kwargs = {kk: np.asarray(vv, dtype=np.float_) for kk, vv in kwargs.items()}

  estimate_, (lb_, ub_), pval_ = test_f(*args, **kwargs)

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(data_vectors, data_vectors, alphas, ddofs)
def test_ztest_from_stats(x, y, alpha, ddof):
  estimate, (lb, ub), pval = wiser.ztest(x, y, alpha=alpha, _ddof=ddof)
  estimate_, (lb_, ub_), pval_ = wiser.ztest_from_stats(
    np.mean(x), np.std(x, ddof=ddof), len(x), np.mean(y), np.std(y, ddof=ddof), len(y), alpha=alpha
  )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(data_vectors, data_vectors, alphas, ddofs)
def test_ztest_coherence(x, y, alpha, ddof):
  general_hyp_tester(wiser.ztest, alpha, x, y, _ddof=ddof)


@given(data_vectors_int, data_vectors_int, alphas, ddofs)
def test_ztest_dtypes(x, y, alpha, ddof):
  general_float_test(wiser.ztest, x, y, alpha=alpha, _ddof=ddof)


@given(data_vector_pairs_2, ddofs)
def test_delta_moments(xx, ddof):
  (x, xp) = xx

  mean_delta, std_delta = wiser._delta_moments(np.mean(xx, axis=1), np.cov(xx, ddof=ddof))

  delta = x - xp
  mean_delta_ = np.mean(delta)
  std_delta_ = np.std(delta, ddof=ddof)

  assert np.isclose(mean_delta, mean_delta_)
  assert np.isclose(std_delta, std_delta_)


@given(train_fracs, integers(2 * wiser.MIN_SPLIT, 100), seeds)
def test_make_train_idx(frac, n, seed):
  random = np.random.RandomState(seed)
  train_idx = wiser._make_train_idx(frac, n, random=random)

  assert train_idx.shape == (n,)
  assert train_idx.dtype.kind == "b"

  assert np.sum(train_idx) >= wiser.MIN_SPLIT
  assert np.sum(~train_idx) >= wiser.MIN_SPLIT

  assert np.sum(train_idx).item() in (
    int(np.ceil(frac * n).item()),
    wiser.MIN_SPLIT,
    n - wiser.MIN_SPLIT,
  )

  # Make sure deterministic
  random = np.random.RandomState(seed)
  train_idx_ = wiser._make_train_idx(frac, n, random=random)

  assert np.all(train_idx == train_idx_)


@given(data_vector_pairs_2, data_vector_pairs_2, alphas, ddofs)
def test_ztest_cv_from_stats(xx, yy, alpha, ddof):
  (x, xp) = xx
  (y, yp) = yy

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate, (lb, ub), pval = wiser.ztest_cv(x, xp, y, yp, alpha=alpha, _ddof=ddof)

  mean1 = np.mean(xx, axis=1)
  cov1 = np.cov(xx, ddof=ddof)
  mean2 = np.mean(yy, axis=1)
  cov2 = np.cov(yy, ddof=ddof)
  estimate_, (lb_, ub_), pval_ = wiser.ztest_cv_from_stats(
    mean1, cov1, len(x), mean2, cov2, len(y), alpha=alpha
  )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(data_vector_pairs_2, data_vector_pairs_2, alphas, ddofs)
def test_ztest_cv_coherence(x, y, alpha, ddof):
  (x, xp) = x
  (y, yp) = y

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    general_hyp_tester(wiser.ztest_cv, alpha, x, xp, y, yp, _ddof=ddof)


@given(data_vector_pairs_2_int, data_vector_pairs_2_int, alphas, ddofs)
def test_ztest_cv_dtypes(x, y, alpha, ddof):
  (x, xp) = x
  (y, yp) = y

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    general_float_test(wiser.ztest_cv, x, xp, y, yp, alpha=alpha, _ddof=ddof)


@given(data_vector_pairs_4, data_vector_pairs_4, alphas, seeds, train_fracs, ddofs)
def test_ztest_cv_train(xx, yy, alpha, seed, train_frac, ddof):
  (x, xp) = xx
  (y, yp) = yy

  random = np.random.RandomState(seed)
  train_idx_x = wiser._make_train_idx(train_frac, len(x), random=random)
  train_idx_y = wiser._make_train_idx(train_frac, len(y), random=random)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate, (lb, ub), pval = wiser.ztest_cv(
      x[~train_idx_x], xp[~train_idx_x], y[~train_idx_y], yp[~train_idx_y], alpha=alpha, _ddof=ddof
    )

  random = np.random.RandomState(seed)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate_, (lb_, ub_), pval_ = wiser.ztest_cv_train(
      x, xp[:, None], y, yp[:, None], alpha=alpha, train_frac=train_frac, random=random, _ddof=ddof
    )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(
  lists(
    gufunc_args("(n,2)->()", dtype=np.float_, elements=easy_floats, min_side=2, max_side=10),
    min_size=1,
    max_size=5,
  )
)
def test_pool_moments(xx):
  # We should test covariances of any d, but this is a private func, so testing only d = 2 is ok
  k_fold = len(xx)

  mean_x = np.zeros((k_fold, 2))
  cov_x = np.zeros((k_fold, 2, 2))
  nobs_x = np.zeros(k_fold)
  for kk in range(k_fold):
    x, = xx[kk]
    nobs_x[kk], d = x.shape
    assert d == 2
    mean_x[kk, :] = np.mean(x, axis=0)
    cov_x[kk, :, :] = np.cov(x.T, ddof=0)

  mean_0, cov_0, nobs_0 = wiser._pool_moments(mean_x, cov_x, nobs_x)

  xx = np.concatenate([x for x, in xx], axis=0)
  nobs_1, d = xx.shape
  assert d == 2
  mean_1 = np.mean(xx, axis=0)
  cov_1 = np.cov(xx.T, ddof=0)

  assert nobs_0 == nobs_1
  assert np.allclose(mean_0, mean_1)
  assert np.allclose(cov_0, cov_1)


@given(integers(1, 100), integers(1, 100), seeds)
def test_fold_idx(n, k, seed):
  k, n = sorted([k, n])  # enforce n <= k

  random = np.random.RandomState(seed)
  idx = wiser._fold_idx(n, k, random=random)

  random = np.random.RandomState(seed)
  idx_ = wiser._fold_idx(n, k, random=random)

  assert np.all(idx == idx_)  # Ensure deterministic

  assert set(idx.tolist()) == set(range(k))


@given(data_vector_pairs_4, data_vector_pairs_4, alphas, folds, seeds)
def test_ztest_stacked_from_stats(xx, yy, alpha, k_fold, seed):
  (x, xp) = xx
  (y, yp) = yy

  assume(len(x) >= wiser.MIN_SPLIT * k_fold)
  assume(len(y) >= wiser.MIN_SPLIT * k_fold)

  random = np.random.RandomState(seed)
  fx = wiser._fold_idx(len(x), k_fold, random=random)
  fy = wiser._fold_idx(len(y), k_fold, random=random)

  mean_x = np.zeros((k_fold, 2))
  cov_x = np.zeros((k_fold, 2, 2))
  nobs_x = np.zeros(k_fold)
  for kk in range(k_fold):
    nobs_x[kk] = np.sum(fx == kk)
    mean_x[kk, :] = np.mean([x[fx == kk], xp[fx == kk]], axis=1)
    cov_x[kk, :, :] = np.cov([x[fx == kk], xp[fx == kk]], ddof=0)

  mean_y = np.zeros((k_fold, 2))
  cov_y = np.zeros((k_fold, 2, 2))
  nobs_y = np.zeros(k_fold)
  for kk in range(k_fold):
    nobs_y[kk] = np.sum(fy == kk)
    mean_y[kk, :] = np.mean([y[fy == kk], yp[fy == kk]], axis=1)
    cov_y[kk, :, :] = np.cov([y[fy == kk], yp[fy == kk]], ddof=0)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate, (lb, ub), pval = wiser.ztest_stacked(x, xp, fx, y, yp, fy, alpha=alpha)

  estimate_, (lb_, ub_), pval_ = wiser.ztest_stacked_from_stats(
    mean_x, cov_x, nobs_x, mean_y, cov_y, nobs_y, alpha=alpha
  )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(data_vector_pairs_2, data_vector_pairs_2, alphas, folds, seeds)
def test_ztest_stacked_coherence(x, y, alpha, k_fold, seed):
  (x, xp) = x
  (y, yp) = y

  assume(len(x) >= k_fold)
  assume(len(y) >= k_fold)

  random = np.random.RandomState(seed)
  fx = wiser._fold_idx(len(x), k_fold, random=random)
  fy = wiser._fold_idx(len(y), k_fold, random=random)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    general_hyp_tester(wiser.ztest_stacked, alpha, x, xp, fx, y, yp, fy)


@given(data_vector_pairs_2_int, data_vector_pairs_2_int, alphas, folds, seeds)
def test_ztest_stacked_dtypes(x, y, alpha, k_fold, seed):
  (x, xp) = x
  (y, yp) = y

  assume(len(x) >= k_fold)
  assume(len(y) >= k_fold)

  random = np.random.RandomState(seed)
  fx = wiser._fold_idx(len(x), k_fold, random=random)
  fy = wiser._fold_idx(len(y), k_fold, random=random)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    general_float_test(wiser.ztest_stacked, x, xp, fx, y, yp, fy, alpha=alpha)


@given(data_vector_pairs_4, data_vector_pairs_4, alphas, folds, seeds)
def test_ztest_stacked_train(xx, yy, alpha, k_fold, seed):
  (x, xp) = xx
  (y, yp) = yy

  assume(len(x) >= wiser.MIN_SPLIT * k_fold)
  assume(len(y) >= wiser.MIN_SPLIT * k_fold)

  # Right now fold idx is not used by stacked estimator so we can pass in None, but might change
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate, (lb, ub), pval = wiser.ztest_stacked(x, xp, None, y, yp, None, alpha=alpha)

  random = np.random.RandomState(seed)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate_, (lb_, ub_), pval_ = wiser.ztest_stacked_train(
      x, xp[:, None], y, yp[:, None], alpha=alpha, k_fold=k_fold, random=random
    )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(data_vector_pairs_4, data_vector_pairs_4, alphas, folds, seeds)
def test_ztest_stacked_train_fast(xx, yy, alpha, k_fold, seed):
  (x, xp) = xx
  (y, yp) = yy

  assume(len(x) >= wiser.MIN_SPLIT * k_fold)
  assume(len(y) >= wiser.MIN_SPLIT * k_fold)

  # Right now fold idx is not used by stacked estimator so we can pass in None, but might change
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate, (lb, ub), pval = wiser.ztest_stacked(x, xp, None, y, yp, None, alpha=alpha)

  random = np.random.RandomState(seed)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate_, (lb_, ub_), pval_ = wiser.ztest_stacked_train_fast(
      x, xp[:, None], y, yp[:, None], alpha=alpha, k_fold=k_fold, random=random
    )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(
  gufunc_args(
    "(n),(n,d),(m),(m,d)->()",
    dtype=np.float_,
    elements=easy_floats,
    min_side={"n": 2 * wiser.MIN_SPLIT, "m": 2 * wiser.MIN_SPLIT, "d": 1},
    max_side={"n": 20, "m": 20, "d": 3},
  ),
  alphas,
  folds,
  seeds,
)
def test_ztest_stacked_train_to_raw(data, alpha, k_fold, seed):
  (x, x_covariates, y, y_covariates) = data

  n_x = len(x)
  n_y = len(y)

  assume(n_x >= wiser.MIN_SPLIT * k_fold)
  assume(n_y >= wiser.MIN_SPLIT * k_fold)

  clf = LinearRegression()

  random = np.random.RandomState(seed)
  fold_idx_x = wiser._fold_idx(n_x, k_fold, random=random)
  fold_idx_y = wiser._fold_idx(n_y, k_fold, random=random)

  xx = []
  yy = []
  xp = []
  yp = []
  for kk in range(k_fold):
    z_covariates = np.concatenate(
      (x_covariates[fold_idx_x != kk, :], y_covariates[fold_idx_y != kk, :]), axis=0
    )
    z = np.concatenate((x[fold_idx_x != kk], y[fold_idx_y != kk]), axis=0)
    clf.fit(z_covariates, z)

    xx.append(x[fold_idx_x == kk])
    yy.append(y[fold_idx_y == kk])
    xp.append(clf.predict(x_covariates[fold_idx_x == kk, :]))
    yp.append(clf.predict(y_covariates[fold_idx_y == kk, :]))

  xx = np.concatenate(xx, axis=0)
  xp = np.concatenate(xp, axis=0)
  yy = np.concatenate(yy, axis=0)
  yp = np.concatenate(yp, axis=0)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate, (lb, ub), pval = wiser.ztest_stacked(
      xx, xp, fold_idx_x, yy, yp, fold_idx_y, alpha=alpha
    )

  random = np.random.RandomState(seed)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate_, (lb_, ub_), pval_ = wiser.ztest_stacked_train(
      x,
      x_covariates,
      y,
      y_covariates,
      alpha=alpha,
      k_fold=k_fold,
      random=random,
      clf=LinearRegression(),
    )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(
  gufunc_args(
    "(n),(n,d),(m),(m,d)->()",
    dtype=np.float_,
    elements=easy_floats,
    min_side={"n": 2 * wiser.MIN_SPLIT, "m": 2 * wiser.MIN_SPLIT, "d": 1},
    max_side={"n": 20, "m": 20, "d": 3},
  ),
  alphas,
  folds,
  seeds,
)
def test_ztest_stacked_train_to_from_stats(data, alpha, k_fold, seed):
  (x, x_covariates, y, y_covariates) = data

  n_x = len(x)
  n_y = len(y)

  assume(n_x >= wiser.MIN_SPLIT * k_fold)
  assume(n_y >= wiser.MIN_SPLIT * k_fold)

  clf = Ridge(alpha=1.0, solver="svd")

  random = np.random.RandomState(seed)
  fold_idx_x = wiser._fold_idx(n_x, k_fold, random=random)
  fold_idx_y = wiser._fold_idx(n_y, k_fold, random=random)

  mean1 = np.zeros((k_fold, 2))
  cov1 = np.zeros((k_fold, 2, 2))
  nobs1 = np.zeros(k_fold)
  mean2 = np.zeros((k_fold, 2))
  cov2 = np.zeros((k_fold, 2, 2))
  nobs2 = np.zeros(k_fold)
  for kk in range(k_fold):
    z_covariates = np.concatenate(
      (x_covariates[fold_idx_x != kk, :], y_covariates[fold_idx_y != kk, :]), axis=0
    )
    z = np.concatenate((x[fold_idx_x != kk], y[fold_idx_y != kk]), axis=0)
    clf.fit(z_covariates, z)

    x_kk = x[fold_idx_x == kk]
    xp = clf.predict(x_covariates[fold_idx_x == kk])
    mean1[kk, 0] = np.mean(x_kk)
    mean1[kk, 1] = np.mean(xp)
    cov1[kk, :, :] = np.cov((x_kk, xp), ddof=0)
    nobs1[kk] = np.sum(fold_idx_x == kk)

    y_kk = y[fold_idx_y == kk]
    yp = clf.predict(y_covariates[fold_idx_y == kk])
    mean2[kk, 0] = np.mean(y_kk)
    mean2[kk, 1] = np.mean(yp)
    cov2[kk, :, :] = np.cov((y_kk, yp), ddof=0)
    nobs2[kk] = np.sum(fold_idx_y == kk)

  estimate, (lb, ub), pval = wiser.ztest_stacked_from_stats(
    mean1, cov1, nobs1, mean2, cov2, nobs2, alpha=alpha
  )

  random = np.random.RandomState(seed)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate_, (lb_, ub_), pval_ = wiser.ztest_stacked_train(
      x,
      x_covariates,
      y,
      y_covariates,
      alpha=alpha,
      k_fold=k_fold,
      random=random,
      clf=Ridge(alpha=1.0, solver="svd"),
    )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(
  gufunc_args(
    "(n),(n,d),(m),(m,d)->()",
    dtype=np.float_,
    elements=easy_floats,
    min_side={"n": 2 * wiser.MIN_SPLIT, "m": 2 * wiser.MIN_SPLIT, "d": 1},
    max_side={"n": 20, "m": 20, "d": 3},
  ),
  alphas,
  folds,
  seeds,
)
def test_ztest_stacked_train_blocks(data, alpha, k_fold, seed):
  (x, x_covariates, y, y_covariates) = data

  n_x = len(x)
  n_y = len(y)

  assume(n_x >= wiser.MIN_SPLIT * k_fold)
  assume(n_y >= wiser.MIN_SPLIT * k_fold)

  clf = Ridge(alpha=1.0, solver="svd")

  random = np.random.RandomState(seed)
  fold_idx_x = wiser._fold_idx(n_x, k_fold, random=random)
  fold_idx_y = wiser._fold_idx(n_y, k_fold, random=random)

  def data_gen(kk):
    R = (
      x[fold_idx_x == kk],
      x_covariates[fold_idx_x == kk, :],
      y[fold_idx_y == kk],
      y_covariates[fold_idx_y == kk, :],
    )
    return R

  data_iter = [partial(data_gen, kk=kk) for kk in range(k_fold)]

  estimate, (lb, ub), pval = wiser._ztest_stacked_train_blocks(data_iter, alpha=alpha, clf=clf)

  random = np.random.RandomState(seed)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    estimate_, (lb_, ub_), pval_ = wiser.ztest_stacked_train_fast(
      x,
      x_covariates,
      y,
      y_covariates,
      alpha=alpha,
      k_fold=k_fold,
      random=random,
      clf=Ridge(alpha=1.0, solver="svd"),
    )

  assert np.isclose(estimate, estimate_)
  assert np.isclose(lb, lb_)
  assert np.isclose(ub, ub_)
  assert np.isclose(pval, pval_)


@given(data_vector_pairs_4, data_vector_pairs_4, alphas, folds, seeds)
def test_ztest_stacked_mlrate_train(xx, yy, alpha, k_fold, seed):
  (x, xp) = xx
  (y, yp) = yy

  assume(len(x) >= wiser.MIN_SPLIT * k_fold)
  assume(len(y) >= wiser.MIN_SPLIT * k_fold)

  random = np.random.RandomState(seed)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    general_hyp_tester(
      wiser.ztest_stacked_mlrate_train,
      alpha,
      x,
      xp[:, None],
      y,
      yp[:, None],
      k_fold=k_fold,
      random=random,
    )
