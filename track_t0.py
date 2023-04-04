import numpy as np
import pandas as pd
from skmine.datasets import fetch_ubiq, fetch_canadian_tv
import time

from skmine.periodic.cycles import PeriodicPatternMiner

# for idx, f in enumerate(glob.glob(ubiq_dir + "/*_IS_data.dat")):
#     user_filename = f.split('/')[-1]
#     print("="*60)
#     print("Mining series: ", user_filename)
#     serie = fetch_ubiq(user_filename=user_filename)
#     start = time.time()
#     print("Mined series: ", user_filename, " in ", round(duration), " seconds")
#     duration = time.time() - start
# for USER in ["25_F", "10_M", "21_F"]:  # ['1_M', '14_F', '25_F', '21_F']:
#     # TODO no errror res.loc[80].E for this serie

USER = '25_F'
user_filename = USER + '_ISE_data.dat'
serie = fetch_ubiq(user_filename)

# TODO afficher event du pattern 0 :  "app.twlauncher"
# USER = '21_F'
# serie = fetch_ubiq(user_filename=USER + '_IS_data.dat')
# serie = fetch_canadian_tv()
start = time.time()
pcm = PeriodicPatternMiner().fit(serie, complex=True, auto_time_scale=True)
res = pcm.discover()

duration = round(time.time() - start)
dur = serie.index[-1] - serie.index[0]
duree = dur / 60 if isinstance(dur, np.integer) else round(dur.total_seconds() / 60)

print("=" * 60)
print("USER: ", USER, "_ISE")
print("-> compression: ", round(pcm.cl / pcm.clRonly * 100, 2))
print("-> code length cl: ", round(pcm.cl, 2))
print("-> L:R ", round(pcm.clR / pcm.cl, 2))
print("-> nb R: ", pcm.nbR)
print("-> nb C: ", pcm.nbC)
print("-> nb simple: ", pcm.nb_simple)
print("-> nb event in serie: ", len(serie))
print(f"-> delta S: {duree} min", )
print("-> card event: ", len(pd.unique(serie)))
print("-> time in s: ", duration)

ind = serie.index.astype('int64')
print(*list(zip(map(str, serie.index[:6]), ind[:6]//10**9)), sep='\n')

#
print("Sum Res", res.sum_E.sum())


# res =pcm.discover(dE_sum=False)
#
# pcm.draw_pattern(80)
# pcm.view()

# print(pcm.out_str)
# print("="*60)
# print(pcm.pl_str)
# import pandas as pd
# from datetime import timedelta
#
# start_time = pd.to_datetime("2020-08-01 06:00:00")
# diff_time = [0, 0,  3,   5,   8,   10,  13, 15,  18, 20, 30]
# events =    [ 'c', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'c','c']
# abs_time = [start_time + timedelta(seconds=k) for k in diff_time]
# simple = pd.Series(events, index=abs_time)
# pcm = PeriodicPatternMiner().fit(simple)
# res = pcm.discover()

# c = fetch_canadian_tv()
# # pcm = PeriodicPatternMiner().fit(serie)
# # res = pcm.discover()
# u = fetch_ubiq(user_filename='10_M_ISE_data.dat')
# import datetime as dt
# import pandas as pd
# one_day = 60 * 24  # a day in minutes
# minutes = [0, 0,  one_day - 1, one_day * 2 - 1, one_day * 3, one_day * 4 + 2, one_day * 7]
#
# S = pd.Series("wake up", index=minutes)
#
# start = dt.datetime.strptime("16/04/2020 07:30", "%d/%m/%Y %H:%M")
# S.index = S.index.map(lambda e: start + dt.timedelta(minutes=e))
# S.index = S.index.round("min")  # minutes as the lowest unit of difference
# S[1] = 'coffee'
