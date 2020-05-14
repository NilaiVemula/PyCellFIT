#!/usr/bin/env python
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


with open("README.rst", "r") as readme_file:
    readme = readme_file.read()

# todo: edit to include all dependencies
requirements = ['numpy>=1.18']

setup(
    name='pycellfit',
    version='0.0.1',
    license='MIT',
    author='Nilai Vemula',
    author_email='nilai.r.vemula@vanderbilt.edu',
    description='Python implementation of the CellFIT method of inferring cellular forces',
    long_description=readme,
    url='https://github.com/NilaiVemula/PyCellFIT',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Natural Language :: English"
    ],
    test_suite='tests',
      cmdclass={'test': PyTest},
)
