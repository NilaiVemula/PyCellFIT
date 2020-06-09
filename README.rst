=========
pycellfit
=========

.. image:: https://travis-ci.com/NilaiVemula/pycellfit.svg?branch=master
  :target: https://travis-ci.com/NilaiVemula/pycellfit
.. image:: https://codecov.io/gh/NilaiVemula/pycellfit/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/NilaiVemula/pycellfit
.. image:: https://readthedocs.org/projects/pycellfit/badge/?version=latest
  :target: https://pycellfit.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://badge.fury.io/py/pycellfit.svg
  :target: https://badge.fury.io/py/pycellfit

Project Description
-------------------
**pycellfit**: an open-source Python implementation of the CellFIT method of inferring cellular forces developed by Brodland et al.

**Author**: Nilai Vemula, Vanderbilt University (working under Dr. Shane Hutson, Vanderbilt University)

**Project Goal**: To develop an open-source version of CellFIT, a toolkit for inferring tensions along cell membranes and pressures inside cells based on cell geometries and their curvilinear boundaries. (See [1]_.)

**Project Timeline**: Initial project started in August 2019 with work based off of XJ Xu. This repository was re-made in May 2020 in order to restart repository structure.

**Project Status**: **Early development**

Getting Started
---------------
This project is available on `PyPI <https://pypi.org/project/pycellfit/>`_ and can be installed using pip.

It recommended that users make a virtual environment and install the package as such:

.. code-block:: console

   pip install pycellfit

Full documentation for this package can be found on `readthedocs <https://pycellfit.readthedocs.io/>`_.

Dependencies
^^^^^^^^^^^^
One of the goals of this project is to avoid dependencies that are difficult to install such as GDAL. This project
primarily depends on numpy, scipy, matplotlib, and other common python packages common in scientific computing. A
full list of dependencies is available in the requirements.txt_ file. All dependencies should be automatically
installed when running pip install.

.. _requirements.txt: requirements.txt

Development
-----------
This project is under active development and not ready for public use. The project is built using Travis CI, and all
tests are run with every commit or merge.

Features
--------
This section will include a list of features available in the package and maybe a check-list of things to add...

Examples
--------
A example walk-through of how to use this module is found in quickstart_.

.. _quickstart: tutorials/README.rst

Future Goals
------------
The final implementation of pycellfit will be as a web-app based on the Django framework. See (add link to
django-pycellfit repo).

References
----------
.. [1] Brodland GW, Veldhuis JH, Kim S, Perrone M, Mashburn D, et al. (2014) CellFIT: A Cellular Force-Inference Toolkit Using Curvilinear Cell Boundaries. PLOS ONE 9(6): e99116. https://doi.org/10.1371/journal.pone.0099116

