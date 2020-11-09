============
Contributing
============

Contributions are welcome, and greatly appreciated !
You can contribute in many ways

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/scikit-mine/scikit-mine/issues.

Please use the issue templates when submitting new issues.


Write Notebooks
~~~~~~~~~~~~~~~

scikit-mine could always include more showcase notebooks. We often concentrate
on implementation details and lack of materials to show how useful our algorithms
can be in real-life situations.
Don't hesitate to bring a little more story telling to scikit-mine !!


Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.


Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub projects. Anything listed in the projects is a feature
to be implemented.

You can also look through GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

For more details check the "Inclusion criteria" section below


==================
Inclusion criteria
==================

Scikit-mine is a library for descriptive analysis, and implements pattern mining algorithms.
Only algorithms belonging to this family of algorithms will be accepted.


Now inclusion of new algorithms into scikit-mine follows a certain number of rules

From most to least important:
 - 200+ citations for the main algorithms.
 - The number of patterns used to describe a set of data should be low. For this we promote algorithms based on `MDL`_ .
 - A low number of parameters (usually one or two). This is to encourage reproducible experiments.


A technique that provides a clear improvement (enhanced datastructures, etc ...) on a widely-used method will
also be condidered for inclusion.

.. _MDL: https://en.wikipedia.org/wiki/Minimum_description_length


===================
Development process
===================


1. If you are a first-time contributor:

   * Go to `https://github.com/scikit-mine/scikit-mine
     <https://github.com/scikit-mine/scikit-mine>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer::

      git clone https://github.com/scikit-mine/scikit-mine

   * Change the directory::

      cd scikit-mine

   * Add the official repository::

      git remote add official https://github.com/scikit-mine/scikit-mine

   * Now, you have remote repositories named:

     - ``official``, which refers to the ``scikit-mine`` repository
     - ``origin``, which refers to your personal fork

2. Setup up developer tools

    * create a local environment, using pip or conda

    * run `pip install -r requirements.txt && pip install -r dev_requirements.txt`

    * make sure test are passing by running `make coverage`

3. Develop your contribution

   * Pull the latest changes from official::

      git checkout master
      git pull official master

   * Create a branch for the feature you want to work on. Since the
     branch name will appear in the merge message, use a sensible name
     such as 'periodic-patterns-MDL-v0'::

      git checkout -b periodic-patterns-MDL-V0

   * Don't forget to update the documentation by editing .rst files inside `docs`.
     and running `make docs` and opening `docs/_build/html/index.html` with your favourite browser

   * Commit locally as you progress (``git add`` and ``git commit``)
     We trigger `black <https://github.com/psf/black>`_ automatically before any commit
     (see `.pre-commit-config.yaml`).

4. To submit your contribution:

    * Push your changes back to your fork on GitHub::
      `git push origin periodic-patterns-MDL-V0`

   * Go to GitHub. The new branch will show up with a green Pull Request
     button - click it.

   * Explain your changes or to ask for review.


Test coverage
~~~~~~~~~~~~~

To measure the test coverage, install
`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`__
(using ``pip install pytest-cov``) and then run::

  $ make coverage

This will print a report with one line for each file in `skmine`,
detailing the test coverage::

    Name                                          Stmts   Miss Branch BrPart  Cover   Missing
    -----------------------------------------------------------------------------------------
    skmine/__init__.py                                4      0      0      0   100%
    skmine/base.py                                   46      4     16      2    90%   154, 176, 202->207, 203->206, 206-207


Writing a benchmark
~~~~~~~~~~~~~~~~~~~

While not mandatory for most pull requests, we ask that performance related
PRs include a benchmark in order to clearly depict the use-case that is being
optimized for.
This section mainly refers to the
`airpseed velocity documentation <https://asv.readthedocs.io/en/latest/writing_benchmarks.html>`_.

In this section we will review how to setup the benchmarks,
and three commands ``asv dev``, ``asv run`` and ``asv continuous``.

You should have installed asv when running `pip install -r dev_requirements.txt`.

First of all you should run the command::

  asv machine


To write  benchmark, add a python file in the ``asv_bench`` directory which contains a
a class with one ``setup`` method and at least one method prefixed with ``time_``.

.. note::

    In scikit-mine we use `asv` in a broad manner, i.e not only to mesure
    time and memory consumption. `asv` let us profile custom indicator, which we use for MDL-based methods
    to track compression ratios and make sure we don't hurt the quality of our compression schemes from one
    development to another.


Take for example the ``slim`` benchmark:

.. code-block:: python

    from skmine.itemsets import SLIM
    from skmine.datasets import make_transactions
    from skmine.preprocessing import TransactionEncoder

    class SLIMBench:
        params = ([20, 1000], [0.3, 0.7])
        param_names = ["n_transactions", "density"]
        # timeout = 20  # timeout for a single run, in seconds
        repeat = (1, 3, 20.0)
        processes = 1

        def setup(self, n_transactions, density):
            transactions = make_transactions(
                n_transactions=n_transactions,
                density=density,
                random_state=7,
            )
            self.transactions = TransactionEncoder().fit_transform(transactions)
            self.slim = SLIM()

        def time_fit(self, *args):
            self.slim.fit(self.transactions)

        def track_data_size(self, *args):
            return self.slim.data_size_

Testing the benchmarks locally
==============================

Prior to running the true benchmark, it is often worthwhile to test that the
code is free of typos. To do so, you may use the command::

  asv dev -b slim

Where the ``SLIM`` above will be run once in your current environment
to test that everything is in order.

Comparing results to master
===========================

Often, the goal of a PR is to compare the results of the modifications in terms
speed to a snapshot of the code that is in the master branch of the
``scikit-mine`` repository. The command ``asv continuous`` is of help here::

    $asv continuous master -b slim
    · Creating environments
    · Discovering benchmarks
    ·· Uninstalling from conda-py3.6-pandas1.0.3
    ·· Building f353431e <v0.0.2> for conda-py3.6-pandas1.0.3.
    ·· Installing f353431e <v0.0.2> into conda-py3.6-pandas1.0.3
    · Running 6 total benchmarks (2 commits * 1 environments * 3 benchmarks)
    [  0.00%] · For scikit-mine commit f353431e <v0.0.2> (round 1/1):
    [  0.00%] ·· Benchmarking conda-py3.6-pandas1.0.3
    [ 16.67%] ··· slim.SLIMBench.time_fit                                                                                  ok
    [ 16.67%] ··· ================ ============= ============ ============= ============
                --                                 density / pruning
                ---------------- -----------------------------------------------------
                n_transactions   0.4 / False   0.4 / True   0.6 / False   0.6 / True
                ================ ============= ============ ============= ============
                        20          329±0.03ms    619±3ms       371±2ms      1.21±0s
                ================ ============= ============ ============= ============
