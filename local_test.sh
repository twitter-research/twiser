#!/bin/bash
# Copyright 2021 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

#
# For now, just run the regular tests, that's it

set -ex
set -o pipefail

TEST_FILE=$1

PY=python3.7

echo "Testing with tests/$TEST_FILE"

# Test using pinned
! test -d env
virtualenv env --python=$PY
source ./env/bin/activate
python --version
pip install -r requirements/base.txt
pip install -r requirements/tests.txt
pip install -e .[test]
pytest tests/$TEST_FILE -s -v --disable-pytest-warnings --hypothesis-seed=0 --cov=twiser --cov-report html
deactivate

# Test using latest
! test -d env_latest
virtualenv env_latest --python=$PY
source ./env_latest/bin/activate
python --version
pip install -e .[test]
pytest tests/$TEST_FILE -s -v --disable-pytest-warnings --hypothesis-seed=0
deactivate
