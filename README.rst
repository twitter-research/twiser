***************
Getting Started
***************

The Twiser package implements a Python library for variance reduction in A/B tests using pre-experiment covariates supporting publication [1]_.
These functions extend the idea of using pre-experiment data for variance reduction previously proposed in publication [2]_.

Installation
============

Only ``Python>=3.7`` is officially supported, but older versions of Python likely work as well.

The package is installed with:

.. code-block:: bash

   pip install twiser

See `GitHub <https://github.com/twitter-research/twiser>`_, `PyPI <https://pypi.org/project/twiser/>`_, and `Read the Docs <https://twiser.readthedocs.io/en/latest/>`_.

Example Usage
=============

A full demo notebook of the package is given in ``demo/survey_loan.ipynb``.
Here is a snippet of the different methods from the notebook:

Setup a predictor as a control variate
--------------------------------------

First, we need to define a regression model.
We can use anything that fits the sklearn idiom of ``fit`` and ``predict`` methods.
This predictor is used to take the ``n x d`` array of treatment unit covariates ``x_covariates`` and predict the treatment outcomes ``n``-length outcome array ``x``.
Likewise, it makes predictions from the ``m x d`` array of control unit covariates ``y_covariates`` to the control ``m``-length outcome array ``y``.

.. code:: python3

    predictor = RandomForestRegressor(criterion="squared_error", random_state=0)

Basic z-test
--------------------

First, we apply the basic two-sample z-test included in Twiser.
This works basically the same as ``scipy.stats.ttest_ind``.

.. code:: python3

    estimate, (lb, ub), pval = twiser.ztest(x, y, alpha=0.05)
    show_output(estimate, (lb, ub), pval)


.. parsed-literal::

    ATE estimate: 0.80 in (-0.14, 1.75), CI width of 1.89, p = 0.0954


Variance reduction with held out data
-------------------------------------

Next, we apply variance reduction where the predictor was trained on a
held out 30% of the data. This is the easiest to show validity, but some
of the added power is lost because not all data is used in the test.

.. code:: python3

    estimate, (lb, ub), pval = twiser.ztest_held_out_train(
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

.. code:: python3

    estimate, (lb, ub), pval = twiser.ztest_cross_val_train(
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

.. code:: python3

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

Other interfaces
----------------

It is also possible to call these methods using raw control predictions instead of training the predictor in the Twiser method.
It also supports a sufficient statistics interface for working with large datasets.
See the `documentation <https://twiser.readthedocs.io/en/latest/>`_ for details.

Support
=======

Create a `new issue <https://github.com/twitter-research/twiser/issues/new/choose>`_.

Links
=====

The `source <https://github.com/twitter-research/twiser>`_ is hosted on GitHub.

The `documentation <https://twiser.readthedocs.io/en/latest/>`_ is hosted at Read the Docs.

Installable from `PyPI <https://pypi.org/project/twiser/>`_.

References
==========

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

License
=======

This project is licensed under the Apache 2 License - see the LICENSE file for details.
