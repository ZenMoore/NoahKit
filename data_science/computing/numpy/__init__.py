import numpy as np

'''
learning material : https://www.runoob.com/numpy/numpy-tutorial.html
api : https://numpy.org/doc/stable/reference/#numpy-reference
'''

'create data (ndarray)'
data = np.array([[1,2,3], [4,5,6]])
np.empty([2, 3])
np.ones((2, 2))
np.zeros([3, 4])
np.asarray([[1,2,3], [4,5,6]])  # lack two params compared with np.array()
np.frombuffer(b'hello world', dtype='S1')  # ['H' 'e' 'l' 'l' 'o' ' ' 'W' 'o' 'r' 'l' 'd'] count=-1:all, offset:start from where
np.fromiter(iter(range(5)), dtype='i1')
np.arange(5)  # like range(5)
np.linspace(0, 10, 10,  endpoint=False, retstep=True) # endpoint=True:contain stop, retstep:show step
np.logspace(0, 10, 10, endpoint=False, base=np.e)


'create dtype'
dtype = np.dtype(np.int32)
dtype = np.dtype('i4')  # int8, int16, int32, int64 = 'i1', 'i2','i4','i8'
dtype = np.dtype([('age', np.int32)])  # create new type [('age', np.int32)]
a = np.array([(10, ), (20, ), (30, )], dtype=dtype)
print(a['age'])
# b:bool, i:int, u:unsigned, f:float, c:complex, V:void
# m:timedelta, M:datetime, O:object, S:string, a:byte-string, U:unicode
dtype = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
a = np.array([('name1', 18, 98), ('name2', 19, 59), ('name3', 20, 60)], dtype=dtype)
print(a)


'ndarray attribute'
# ndim:dimension, shape:n1 x n2 x ..., itemsize:element size
# flags:memory type:C_COUTIGUOUS, F_COUTINUOUS, OWNDATA, WRITABLE, ALIGNED, UPDATEIFCOPY
a = np.array([[1,2,3],[4,5,6]])
a.shape = (3, 2)  # change shape, or b = a.reshape(3, 2), but reshape returns non-copy duplication
print(a)

'slice and index'
data = np.arange(10)
slice = slice(2, 7, 2) # from 2 to 7, step is 2
print(data[slice])
print(data[2:7:2])
data = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(data[1,:])
print(data[1,...])
print(data[[0, 1, 2], [0, 1, 0]])  # (0, 0), (1, 1), (2, 0)
rows = [[0, 0], [2, 2]]  #    v      v         v       v
cols = [[0, 2], [0, 2]]  #      v       v         v       v
print(data[rows, cols])  # [[(0,0), (0, 2)], [(2, 0), (2, 2)]]
print(data[data > 3])
a = np.array([np.nan,  1,2,np.nan,3,4,5])
print (a[~np.isnan(a)])
x=np.arange(32).reshape((8,4))
print (x[[4,2,1,7]])
x=np.arange(32).reshape((8,4))
print(x[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])

'broadcast'
# no broadcast : a.shape = b.shape -> element-wise operation
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
print(a * b)

# with broadcast : a.shape != b.shape
a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([1,2,3])
print(a + b)
# a same
b = np.tile(np.array([1, 2, 3]), (4, 1))  # copy 4 rows, 1 col
print(a + b)

'iteration'
a = np.arange(6).reshape(2,3)
for x in np.nditer(a):
    print(x, end=', ')  # C-order = row-first
print()
for x in np.nditer(a.T):
    print(x, end=', ')  # C-order = row-first
print()
for x in np.nditer(a.T.copy(order = 'C')):
    print(x, end = ', ')
print()
for x in np.nditer(a, order='F'):
    print(x, end=', ')
print()
for x in np.nditer(a, op_flags=['readwrite']):  # 'write-only'
    x += 1
    print(x, end=', ')
print()
for x in np.nditer(a, order='F', flags=['external_loop']):  # c_index, f_index, multi_index
    print(x, end=', ')
print()
b = np.array([1,  2,  3], dtype =  int)
for x,y in np.nditer((a,b)):
    print ("%d:%d"  %  (x,y), end=", " )
print()
for x,y in np.nditer([a,b]):
    print ("%d:%d"  %  (x,y), end=", " )
print()

'array operation'
# change shape
a = np.arange(8).reshape(2,4)
print(a)
for x in a.flat:  # one-dim iterator, non-copy
    print(x, end=', ')
print()
for x in a.flatten():  # one-dim iterator, copy
    print(x, end=', ')
print()
print(a.ravel(order='C'))  # not iterator

# overturn
a = np.arange(8).reshape(2,4)
print(a.transpose())  # copy
print(a.T)
a = np.arange(8).reshape(2,2,2)
print(np.rollaxis(a, 2, 0))  # fuck.
print(np.swapaxes(a, 2, 0))  # difference : rotation v.s. reconstruction

# change dimension
x = np.array([[1], [2], [3]])
y = np.array([4, 5, 6])
b = np.broadcast(x, y)  # broadcast x to y, return iterators
print(b)
r, c = b.iters
print(next(r), next(c))
print(next(r), next(c))
print(next(r), next(c))
print(next(r), next(c))
print(next(r), next(c))
print(next(r), next(c))
print(next(r), next(c))
print(next(r), next(c))

a = np.arange(4).reshape(1,4)
print(np.broadcast_to(a, (4, 4)))  # change a(1, 4) to (4, 4)

x = np.array(([1,2],[3,4]))  # (2, 2)
print(np.expand_dims(x, axis=0).shape)  # (1, 2, 2)
b = np.expand_dims(x, axis=1)
print(b.shape)  # (2, 1, 2)
print(np.squeeze(b).shape)  # delete =1 axis

# concatenation
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.concatenate([a, b], axis=0))  # along the existing axis
print(np.stack((a, b)))  # create a new axis
print(np.hstack((a, b)))
print(np.vstack((a, b)))

# split
a = np.arange(9)
print(np.split(a, 3))
print(np.split(a, [4, 7]))
a = np.arange(16).reshape(4, 4)
print(np.split(a, 2, 1))
print(np.hsplit(a, 2))
print(np.vsplit(a, 2))

# addition and deletion
a = np.arange(6).reshape(2, 3)
print(a)
print(np.reshape(a, (3, 2)))
print(np.resize(a, (3, 2)))  # all these three is copy
print(np.resize(a, (3, 3)))  # complete automatically and repetitively
print(a)  # unchanged
print(np.append(a, [7, 8, 9]))  # default axis->always return one-dim
print((np.append(a, [[7, 8, 9]], axis=0)))
print((np.append(a, [[7, 8, 9], [10, 11, 12]], axis=1)))
a = np.array([[1,2],[3,4],[5,6]])
print(np.insert(a, 3, 11))
print(np.insert(a, 1, [11], axis=0))  # broadcast
print(np.insert(a, 1, 11, axis=1))
a = np.arange(12).reshape(3,4)
print(np.delete(a, 5))  # always one-dim
print(np.delete(a, 1, axis=1))  # delete 2nd col
a = np.array([1,2,3,4,5,6,7,8,9,10])
print (np.delete(a, np.s_[::2]))
a = np.array([5,2,6,2,7,5,6,8,2,9])
print(np.unique(a))  # delete duplicated elements


'bit operation'
a = np.binary_repr(242, width=8)
print(np.bitwise_and(13, 17))
# invert, bitwise_xx, left_shift, right_shift

'functions'
# string
print(np.char.add(['hello', 'hi'],[' abc', ' xyz']))
print(np.char.multiply('Runoob ',3))
print(np.char.center('Runoob', 20,fillchar = '*'))
print (np.char.capitalize('runoob'))
print (np.char.title('i like runoob'))
print (np.char.lower('RUNOOB'))  # can be array
print (np.char.upper('runoob'))
print (np.char.split ('www.runoob.com', sep = '.'))
print (np.char.splitlines('i\rlike runoob?'))
print (np.char.strip(['arunooba','admin','java'],'a'))
print (np.char.join([':','-'],['runoob','google']))  # can be single string
print (np.char.replace ('i like runoob', 'oo', 'cc'))
a = np.char.encode('runoob', 'cp500')
print (np.char.decode(a,'cp500'))

# math
# triangles, degrees
# around, floor, ceil
# add, subtract, multiply, divide, reciprocal, power, mod,

# statistics
# amin, amax, mean, var, std, median, mode, ptp, percentile
# pay attention to axis, keep_dim

# sorting
# sort, argsort, lexsort:multi-seq sorting
# msort=sort(axis=0), sort_complex
nm =  ('raju','anil','ravi','amar')
dv =  ('f.y.',  's.y.',  's.y.',  'f.y.')
ind = np.lexsort((dv,nm))
print([nm[i]  +  ", "  + dv[i]  for i in ind])
a = np.array([3, 4, 2, 1])
print(np.partition(a, 3))  # sort from smaller to larger，former is larger than 3, latter is smaller than 3
print(np.argpartition(a, 2))  # 2 is index
print(a[np.argpartition(a, 2)[2]])  # return 3-smallest
print(a[np.argpartition(a, -2)[-2]])  # return 2-largest
print(a[np.argpartition(a, [2,3])[2]])
# argmin, argmax

# filtrate
# nonzero
x = np.arange(9.).reshape(3,  3)
print(np.where(x > 3))
print(x[np.where(x > 3)])
print(np.extract(x > 3, x))  #  same as x[np.where(x > 3)]


'deepcopy and copy:https://www.runoob.com/numpy/numpy-copies-and-views.html'

'matrix'
a = np.matrix('1, 2 ; 3, 4')
print(a)
b = np.asarray(a)  # matrix to ndarray
print(b)
print(np.asmatrix(b))
# np.matlib.zeros (ones, rand, identity, empty, eye)

'linear algebra'
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.dot(a, b))  # dim=1: inner product; dim=2: matrix product; dim>2: dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
print(np.vdot(a, b))  # flat, inner product
print(np.inner(a, b))  # not flat, inner product : [1*11+2*12, 1*13+2*14; 3*11+4*12, 3*13+4*14]
print(np.matmul(a, b))  # same dim: matrix product. different dim: broadcast
print(np.linalg.det(a))
print(np.linalg.solve(np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]]), np.array([[6], [-4], [27]])))
print(np.linalg.inv(a))

'IO'
a = np.array([1,2,3,4,5])
file = 'path-to-npy.npy'
np.save(file, a)
b = np.load(file)
np.savez(file, a, name_b = b)  # to save more in one file .npz, a is arr_0 as default, b is name_b
# savetxt, loadtxt
a=np.arange(0,10,0.5).reshape(4,-1)
np.savetxt("out.txt",a,fmt="%d",delimiter=",") # save as integer, split by comma
b = np.loadtxt("out.txt",delimiter=",") # when load, split by comma, too




