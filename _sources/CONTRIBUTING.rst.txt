============
Contributing
============

Contributions are welcome, and greatly appreciated !
You can contribute in many ways:

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