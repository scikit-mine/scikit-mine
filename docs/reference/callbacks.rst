.. _callbacks:

Callback API
************

scikit-mine allows to define custom callbacks to **track metrics and artifacts
during exploration**.
The definition of callbacks is volontarily very permissive so that
it will work in most cases.

Simply put, a callback is a method to be called after the execution of a method it targets.
**This allows tracking models attributes (accesing the model directly via the ``self`` keyword),
or function results**.

.. autoclass:: skmine.callbacks.CallBacks

.. autofunction:: skmine.callbacks.mdl_prints