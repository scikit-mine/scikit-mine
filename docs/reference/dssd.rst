.. _dssd:

DSSD
****

Miner, Selection Strategies, Quality Measures, Refinement operators to mine diverse subgroup sets


Miner 
-------
.. autoclass:: skmine.dssd.DSSDMiner


Selection strategies
--------------------
.. autoclass:: skmine.dssd.SelectionStrategy
.. autoclass:: skmine.dssd.Desc
.. autoclass:: skmine.dssd.VarDescFast
.. autoclass:: skmine.dssd.VarDescStandard
.. autoclass:: skmine.dssd.Cover
.. autoclass:: skmine.dssd.VarCover
.. automethod:: skmine.dssd.multiplicative_weighted_covering_score_smart


Quality measures
----------------
.. autoclass:: skmine.dssd.QualityMeasure
.. autoclass:: skmine.dssd.WRACC
.. autoclass:: skmine.dssd.KL
.. autoclass:: skmine.dssd.WKL
.. autoclass:: skmine.dssd.TSQuality
.. autoclass:: skmine.dssd.EuclideanEub
.. autoclass:: skmine.dssd.DtwDba
.. autoclass:: skmine.dssd.FastDtwDba


Refinement operators
--------------------
.. autoclass:: skmine.dssd.RefinementOperator
.. autoclass:: skmine.dssd.RefinementOperatorImpl


Subgroup, Condition, Description
--------------------------------
.. autoclass:: skmine.dssd.Subgroup
.. autoclass:: skmine.dssd.Cond
.. autoclass:: skmine.dssd.Description
.. automethod:: skmine.dssd.apply_dominance_pruning
.. automethod:: skmine.dssd.update_topk
.. autoattribute:: skmine.dssd.FuncCover
.. autoattribute:: skmine.dssd.FuncQuality
.. autoattribute:: skmine.dssd.ColumnShares




