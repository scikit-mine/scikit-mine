from skmine.itemsets import SLIM

from skmine.datasets.fimi import fetch_any

import json

data_sets = [fetch_any(f"{_}.dat") for _ in ("chess", "mushroom", "connect")]

res = dict()
for data in data_sets:
    slim = SLIM()
    slim.fit(data)
    res.update({
        f"slim_data_size_{data.name}" : slim.data_size_,
        f"slim_model_size_{data.name}" : slim.model_size_,
    })
print(json.dumps(res))
