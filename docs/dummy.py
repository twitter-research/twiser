# Copyright 2021 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sphinx

# import extra deps and use it to keep pipreqs and flake8 happy
for pkg in (sphinx,):
  print("%s %s" % (pkg.__name__, pkg.__version__))
