*****************
The WISER Package
*****************

Advanced variance reduction methods.

Installation
============

Only ``Python>=3.6`` is officially supported, but older versions of Python likely work as well.

The package is installed with:

.. code-block:: bash

   pip install wiser

See the `GitHub <https://github.com/twitter/wiser>`_, `PyPI <https://pypi.org/project/wiser/>`_, and `Read the Docs <https://wiser.readthedocs.io/en/latest/>`_.

Example Usage
=============

TODO

Contributing
============

The following instructions have been tested with Python 3.7.4 on Mac OS (11.5.2).

Install in editable mode
------------------------

First, define the variables for the paths we will use:

.. code-block:: bash

   GIT=/path/to/where/you/put/repos
   ENVS=/path/to/where/you/put/virtualenvs

Then clone the repo in your git directory ``$GIT``:

.. code-block:: bash

   cd $GIT
   git clone https://github.com/twitter/wiser.git

Inside your virtual environments folder ``$ENVS``, make the environment:

.. code-block:: bash

   cd $ENVS
   virtualenv wiser --python=python3.7
   source $ENVS/wiser/bin/activate

Now we can install the pip dependencies. Move back into your git directory and run

.. code-block:: bash

   cd $GIT/wiser
   pip install -r requirements/base.txt
   pip install -e .  # Install the package itself

Contributor tools
-----------------

First, we need to setup some needed tools:

.. code-block:: bash

   cd $ENVS
   virtualenv wiser_tools --python=python3.7
   source $ENVS/wiser_tools/bin/activate
   pip install -r $GIT/wiser/requirements/tools.txt

To install the pre-commit hooks for contributing run (in the ``wiser_tools`` environment):

.. code-block:: bash

   cd $GIT/wiser
   pre-commit install

To rebuild the requirements, we can run:

.. code-block:: bash

   cd $GIT/wiser

   # Check if there any discrepancies in the .in files
   pipreqs wiser/ --diff requirements/base.in
   pipreqs tests/ --diff requirements/tests.in
   pipreqs docs/ --diff requirements/docs.in

   # Regenerate the .txt files from .in files
   pip-compile-multi --no-upgrade

Generating the documentation
----------------------------

First setup the environment for building with ``Sphinx``:

.. code-block:: bash

   cd $ENVS
   virtualenv wiser_docs --python=python3.7
   source $ENVS/wiser_docs/bin/activate
   pip install -r $GIT/wiser/requirements/docs.txt

Then we can do the build:

.. code-block:: bash

   cd $GIT/wiser/docs
   make all
   open _build/html/index.html

Documentation will be available in all formats in ``Makefile``. Use ``make html`` to only generate the HTML documentation.

Running the tests
-----------------

The tests for this package can be run with:

.. code-block:: bash

   cd $GIT/wiser
   ./local_test.sh

The script creates an environment using the requirements found in ``requirements/test.txt``.
A code coverage report will also be produced in ``$GIT/wiser/htmlcov/index.html``.

Deployment
----------

The wheel (tar ball) for deployment as a pip installable package can be built using the script:

.. code-block:: bash

   cd $GIT/wiser/
   ./build_wheel.sh

This script will only run if the git repo is clean, i.e., first run ``git clean -x -ff -d``.

Links
=====

The `source <https://github.com/twitter/wiser>`_ is hosted on GitHub.

The `documentation <https://wiser.readthedocs.io/en/latest/>`_ is hosted at Read the Docs.

Installable from `PyPI <https://pypi.org/project/wiser/>`_.

License
=======

This project is licensed under the Apache 2 License - see the LICENSE file for details.
