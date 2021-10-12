from setuptools import find_packages, setup

from wiser import __version__


def read_requirements(name):
  with open("requirements/" + name + ".in") as f:
    requirements = f.read().strip()
  requirements = requirements.replace("==", ">=").splitlines()  # Loosen strict pins
  return [pp for pp in requirements if pp[0].isalnum()]


requirements = read_requirements("base")
test_requirements = read_requirements("tests")

with open("README.rst") as f:
  long_description = f.read()

setup(
  name="wiser",
  version=__version__,
  packages=find_packages(),
  url="https://github.com/twitter/wiser/",
  author="Ryan Turner",
  author_email=("rdturnermtl@github.com"),
  license="Apache v2",
  description="Advanced variance reduction methods.",
  python_requires=">=3.6",
  install_requires=requirements,
  extras_require={"test": test_requirements},
  long_description=long_description,
  long_description_content_type="text/x-rst",
  platforms=["any"],
)
