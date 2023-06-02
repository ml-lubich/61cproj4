from numc import Matrix
import numpy as np
import unittests as ut

def foo(n):
    a = Matrix(n, n, list(range(0, n*n)))
    ma = a * a
    b = np.array([list(range(i, i+n)) for i in range(0, n*n, n)])
    mb = np.matmul(b, b)
    print(ma)
    print(mb)

p = ut.TestPow()
p.test_medium_pow()
p.test_medium_pow()
p.test_medium_pow()
p.test_medium_pow()
p.test_medium_pow()
p.test_large_pow()
p.test_large_pow()
