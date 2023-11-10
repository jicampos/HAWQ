"""Setup script for HAWQ"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import setuptools

with io.open("README.md", "r", encoding="utf8") as fh:
  long_description = fh.read()

setuptools.setup(
    name="hawq",
    version="0.1.0",
    author="",
    author_email="",
    maintainer="",
    maintainer_email="",
    packages=setuptools.find_packages(),
    scripts=[],
    url="https://github.com/Zhen-Dong/HAWQ",
    license="MIT license",
    description="Quantization libary for PyTorch",
    long_description=long_description,
    install_requires=[
        "numpy>=1.16.0",
        "pyparser",
        "setuptools>=41.0.0",
    ],
    extra_requires={
      "dev": ["pytest", "twine"]
    },
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
)
