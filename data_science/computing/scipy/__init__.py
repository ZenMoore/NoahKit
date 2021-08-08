import scipy.optimize as optimize
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.fft as fft
import scipy.integrate as integrate
import scipy.constants as constants

'''
learning material : https://www.runoob.com/scipy/scipy-constants.html
api : https://docs.scipy.org/doc/scipy/reference/cluster.html
'''


'fft : image denoising'
chip = plt.imread('../../asset/img.png')
chip = np.mean(chip, axis=2)
f_chip = fft.fft2(chip)
plt.imshow(chip)
plt.show()
f_chip[f_chip > 3e2] = 0
new_chip = np.real(fft.ifft2(f_chip))
plt.imshow(new_chip)
plt.show()

'integration'
f = lambda x : (1-x**2)**0.5
print(integrate.quad(f, -1, 1))  # area, error
# we can also ODE !!!

'io'
data = np.zeros((2, 3))
io.savemat('path-to-.mat', {'data1': data})
io.loadmat('path-to-.mat')['data1']

'constant'
print(dir(constants))  # see unit_const

'optimization'
# numpy can get the root of linear function or polynomial function
# but cannot get the root of non-linear function : so scipy.optimize
f = lambda x : x + np.cos(x)
print(optimize.root(f, 0))
print(optimize.minimize(f, 0, method='BFGS'))

'sparse matrix'
arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])
print(csr_matrix(arr))  # csc_matrix: compressed by column

arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print(csr_matrix(arr).data)  # non-zero data
print(csr_matrix(arr).count_nonzero())
print(csr_matrix(arr).eliminate_zeros())
print(csr_matrix(arr).sum_duplicates())  # delete duplicated data
print(csr_matrix(arr).tocsc())