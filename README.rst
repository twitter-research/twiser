***************
Getting Started
***************

Advanced variance reduction methods.

Installation
============

Only ``Python>=3.7`` is officially supported, but older versions of Python likely work as well.

The package is installed with:

.. code-block:: bash

   pip install twiser

See `GitHub <https://github.com/twitter/twiser>`_, `PyPI <https://pypi.org/project/twiser/>`_, and `Read the Docs <https://twiser.readthedocs.io/en/latest/>`_.

Example Usage
=============

A full demo notebook of the package is given in ``demo/survey_loan.ipynb``.
Here is a snippet of the different methods from the notebook:

Basic :math:`z`-test
--------------------

First, we apply the basic two-sample :math:`z`-test included in twiser.
This works basically the same as ``scipy.stats.ttest_ind``.

.. code:: ipython3

    estimate, (lb, ub), pval = twiser.ztest(x, y, alpha=0.05)
    show_output(estimate, (lb, ub), pval)


.. parsed-literal::

    ATE estimate: 0.80 in (-0.14, 1.75), CI width of 1.89, p = 0.0954


Variance reduction with held out data
-------------------------------------

Next, we apply variance reduction where the predictor was trained on a
held out 30% of the data. This is the easiest to show validity, but some
of the added power is lost because not all data is used in the test.

.. code:: ipython3

    estimate, (lb, ub), pval = twiser.ztest_cv_train(
      x,
      x_covariates,
      y,
      y_covariates,
      alpha=0.05,
      train_frac=0.3,
      predictor=predictor,
      random=np.random.RandomState(123),
    )
    show_output(estimate, (lb, ub), pval)


.. parsed-literal::

    ATE estimate: 1.40 in (0.20, 2.59), CI width of 2.39, p = 0.0217*


Variance reduction with cross validation
----------------------------------------

To be more statistically efficient we train and predict using 10-fold
cross validation. Here, no data is wasted. As we can see it is a more
significant result.

.. code:: ipython3

    estimate, (lb, ub), pval = twiser.ztest_stacked_train(
      x,
      x_covariates,
      y,
      y_covariates,
      alpha=0.05,
      k_fold=10,
      predictor=predictor,
      random=np.random.RandomState(123),
    )
    show_output(estimate, (lb, ub), pval)


.. parsed-literal::

    ATE estimate: 1.38 in (0.51, 2.25), CI width of 1.74, p = 0.0019*


Variance reduction in-sample
----------------------------

In the literature it is popular to train the predictor in the same
sample as the test. This often gives the most power. However, any
overfitting in the predictor can also invalidate the results.

.. code:: ipython3

    estimate, (lb, ub), pval = twiser.ztest_in_sample_train(
      x,
      x_covariates,
      y,
      y_covariates,
      alpha=0.05,
      predictor=predictor,
      random=np.random.RandomState(123),
    )
    show_output(estimate, (lb, ub), pval)


.. parsed-literal::

    ATE estimate: 0.86 in (0.24, 1.49), CI width of 1.24, p = 0.0065*

Support
=======

Create a `new issue <https://github.com/twitter-research/twiser/issues/new/choose>`_ or `join a discussion <https://github.com/twitter-research/twiser/discussions>`_.

Links
=====

The `source <https://github.com/twitter/twiser>`_ is hosted on GitHub.

The `documentation <https://twiser.readthedocs.io/en/latest/>`_ is hosted at Read the Docs.

Installable from `PyPI <https://pypi.org/project/twiser/>`_.

License
=======

This project is licensed under the Apache 2 License - see the LICENSE file for details.
