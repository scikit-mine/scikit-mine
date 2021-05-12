.. _datasets:

Datasets
********

Methods to fetch, generate, and describe datasets.


Standard datasets for itemset mining
------------------------------------
.. autofunction:: skmine.datasets.fimi.fetch_any

.. autofunction:: skmine.datasets.fimi.fetch_chess

.. autofunction:: skmine.datasets.fimi.fetch_connect

.. autofunction:: skmine.datasets.fimi.fetch_mushroom

.. autofunction:: skmine.datasets.fimi.fetch_pumsb

.. autofunction:: skmine.datasets.fimi.fetch_pumsb_star

.. autofunction:: skmine.datasets.fimi.fetch_kosarak

.. autofunction:: skmine.datasets.fimi.fetch_retail

.. autofunction:: skmine.datasets.fimi.fetch_accidents


Logs datasets for periodic pattern mining
-----------------------------------------
.. autofunction:: skmine.datasets.fetch_health_app

.. autofunction:: skmine.datasets.fetch_canadian_tv


Instacart Market Basket Analysis
--------------------------------
.. autofunction:: skmine.datasets.fetch_instacart


Synthetic data generation
-------------------------

.. autofunction:: skmine.datasets.make_transactions
.. autofunction:: skmine.datasets.make_classification


utils
-------
.. autofunction:: skmine.datasets.utils.describe
.. autofunction:: skmine.datasets.utils.describe_logs
.. autofunction:: skmine.datasets.get_data_home
