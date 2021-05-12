===============================
Compatibility with Scikit-Learn
===============================
`scikit-learn <https://scikit-learn.org/stable/>`_ is the golden standard for general
puprose machine learning. As a rule of thumb, we follow scikit-learn functional definitions.

-----------------

*scikit-learn* is a library for statistical learning, or **machine-learning**.

*scikit-mine*, on its side, is a library for (yet statistical) **pattern mining**.

So what does this change ?
*scikit-mine* gives more attention to discrete values, because **it looks for co-occuring symbols in the data**.
To this purpose, we sometimes need to extend scikit-learn capabilities to tightly integrate the notion
of symbols in our learning processes.


Using data mining methods as feature extraction blocks for Machine Learning
---------------------------------------------------------------------------
The :ref:`feature_extraction` module implements a set of Transformers/Encoders
to get you from raw data to a more advanced, structured kind of data : 
the kind of data that is easily manageable and prone to give you the best performance.

Sometimes *scikit-learn* provides us the tools we exactly need, sometimes not.
**Scikit-mine addresses data ingestion by implementing its own blocks,
which are compatible wtih scikit-learn**.


Pipelines
---------
scikit-mine models are designed for possible integration in `scikit-learn pipelines <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.
This makes possible to build "symbolic classifiers", using scikit-mine pattern encoding schemes
to serve predictions, or just use scikit-mine as the first part of a scikit-learn pipeline, as mentioned
in the previous section.


Other implementation details
----------------------------
We use `joblib <https://joblib.readthedocs.io/en/latest/>`_ as default to parallelise our code.
We also set the *prefer* parameter when instantiating `joblib.Parallel <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_, 
so users don't have to manually choose between threads and processes for optimal execution.
