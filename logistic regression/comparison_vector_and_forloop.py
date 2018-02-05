import numpy as np
import time
a = np.random.rand(1000000)
b = np.random.rand(1000000)

t_begin = time.time()
c = np.dot(a, b)
t_end = time.time()
print("vectorized version: " + str((t_end-t_begin)*1000) + "ms")
print('c: ' + str(c))
t_begin = time.time()
c = 0;
for i in range(1000000):
    c += a[i]*b[i]
t_end = time.time()
print("for loop version: " + str((t_end-t_begin)*1000) + "ms")
print('c: ' + str(c))