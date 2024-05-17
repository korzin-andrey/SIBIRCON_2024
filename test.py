import time
import numpy as np

start_all = time.perf_counter()

for i in range(10000):
    x_rand = np.random.rand(1000)

finish_all = (time.perf_counter() - start_all)

print("Total calculation time: {}".format(finish_all))
