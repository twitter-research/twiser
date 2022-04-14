How to Contribute
=================

We'd love to get patches from you!

Getting Started
---------------

The following instructions have been tested with Python 3.7.4 on Mac OS (11.5.2).

Building dependencies
---------------------

First, define the variables for the paths we will use:

.. code-block:: bash

   GIT=/path/to/where/you/put/repos
   ENVS=/path/to/where/you/put/virtualenvs

Then clone the repo in your git directory ``$GIT``:

.. code-block:: bash

   cd $GIT
   git clone https://github.com/twitter/twiser.git

Inside your virtual environments folder ``$ENVS``, make the environment:

.. code-block:: bash

   cd $ENVS
   virtualenv twiser --python=python3.7
   source $ENVS/twiser/bin/activate

Now we can install the pip dependencies. Move back into your git directory and run

.. code-block:: bash

   cd $GIT/twiser
   pip install -r requirements/base.txt
   pip install -e .  # Install the package itself

Install in editable mode
^^^^^^^^^^^^^^^^^^^^^^^^

First, define the variables for the paths we will use:

.. code-block:: bash

   GIT=/path/to/where/you/put/repos
   ENVS=/path/to/where/you/put/virtualenvs

Then clone the repo in your git directory ``$GIT``:

.. code-block:: bash

   cd $GIT
   git clone https://github.com/twitter/twiser.git

Inside your virtual environments folder ``$ENVS``, make the environment:

.. code-block:: bash

   cd $ENVS
   virtualenv twiser --python=python3.7
   source $ENVS/twiser/bin/activate

Now we can install the pip dependencies. Move back into your git directory and run

.. code-block:: bash

   cd $GIT/twiser
   pip install -r requirements/base.txt
   pip install -e .  # Install the package itself

Contributor tools
^^^^^^^^^^^^^^^^^

First, we need to setup some needed tools:

.. code-block:: bash

   cd $ENVS
   virtualenv twiser_tools --python=python3.7
   source $ENVS/twiser_tools/bin/activate
   pip install -r $GIT/twiser/requirements/tools.txt

To install the pre-commit hooks for contributing run (in the ``twiser_tools`` environment):

.. code-block:: bash

   cd $GIT/twiser
   pre-commit install

To rebuild the requirements, we can run:

.. code-block:: bash

   cd $GIT/twiser

   # Check if there any discrepancies in the .in files
   pipreqs twiser/ --diff requirements/base.in
   pipreqs tests/ --diff requirements/tests.in
   pipreqs docs/ --diff requirements/docs.in

   # Regenerate the .txt files from .in files
   pip-compile-multi --no-upgrade

Building the Project
--------------------

The wheel (tar ball) for deployment as a pip installable package can be built using the script:

.. code-block:: bash

   cd $GIT/twiser/
   ./build_wheel.sh

This script will only run if the git repo is clean, i.e., first run ``git clean -x -ff -d``.

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

First setup the environment for building with ``Sphinx``:

.. code-block:: bash

   cd $ENVS
   virtualenv twiser_docs --python=python3.7
   source $ENVS/twiser_docs/bin/activate
   pip install -r $GIT/twiser/requirements/docs.txt

Then we can do the build:

.. code-block:: bash

   cd $GIT/twiser/docs
   make all
   open _build/html/index.html

Documentation will be available in all formats in ``Makefile``. Use ``make html`` to only generate
the HTML documentation.

Workflow
--------

We follow the `GitHub Flow
Workflow <https://guides.github.com/introduction/flow/>`__, which
typically involves forking the project into your GitHub account, adding
your changes to a feature branch, and opening a Pull Request to
contribute those changes back to the project.

Testing
-------

The tests for this package can be run with:

.. code-block:: bash

   cd $GIT/twiser
   ./local_test.sh

The script creates an environment using the requirements found in ``requirements/test.txt``.
A code coverage report will also be produced in ``$GIT/twiser/htmlcov/index.html``.

Style
-----

The coding style is enforced the the pre-commit hooks in ``.pre-commit-config.yaml``. Most
importantly, just use black. That takes care of most of the formatting.

Issues
------

When filing an issue, try to adhere to the provided template if
applicable. In general, the more information you can provide about
exactly how to reproduce a problem, the easier it will be to help fix
it.

Code of Conduct
===============

We expect all contributors to abide by our `Code of
Conduct <https://github.com/twitter/.github/blob/main/code-of-conduct.md>`__.
