import numpy as np
from numba import vectorize, float64
import time

# функция Python для вычисления гипотенузы
def pythagorean_theorem(a, b):
    return np.sqrt(a**2 + b**2)

# векторизованная функция с Numba
@vectorize([float64(float64, float64)])
def numba_pythagorean_theorem(a, b):
    return np.sqrt(a**2 + b**2)

# создаем большие массивы данных
a = np.array(np.random.sample(1000000), dtype=np.float64)
b = np.array(np.random.sample(1000000), dtype=np.float64)

# измеряем время выполнения для обычной функции
rtp = np.vectorize(pythagorean_theorem)
start_time = time.time()
rtp(a, b)
print("Обычное Python время:", time.time() - start_time)

# измеряем время выполнения для векторизованной функции
start_time = time.time()
numba_pythagorean_theorem(a, b)
print("Numba @vectorize время:", time.time() - start_time)
