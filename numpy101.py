import numpy as np


def array_attribute():
    array = np.array([[1,2,3],[4,5,6]])
    print(array)
    print('number of dim:',array.ndim)
    print('shape:',array.shape)
    print('size:',array.size)
    pass


def array_create():
	a = np.array([2,32,4], dtype=np.int)
	print(a.dtype)
	a = np.array([2,32,4], dtype=np.float64)
	print(a.dtype)
	a = np.array([[1,2,3],[4,5,6]])
	print(a)
	a = np.zeros((3,4))
	print(a)
	a = np.ones((3,4))
	print(a)
	a = np.empty((3,4))
	print(a)
	a = np.arange(12)
	print(a)
	a = np.arange(12).reshape((3,4))
	print(a)
	a = np.linspace(1,10,20)
	print(a)
	pass


def array_basic():
	a = np.array([10,20,30,40])
	b = np.arange(4)
	print(a,b)
	c = a - b
	print(c)
	c = b**2
	print(c)
	c = 10*np.sin(a)
	print(c)
	c = 10*np.cos(a)
	print(c)
	c = 10*np.tan(a)
	print(c)
	print(b<3)

	a = np.array([[1,1],[0,1]])
	b = np.arange(4).reshape((2,2))
	print(a,b)
	c = a*b
	print(c)
	c_dot = np.dot(a,b)
	print(c_dot)
	c_dot2 = a.dot(b)
	print(c_dot2)
	a = np.random.random((2,4))
	print(a)
	print(np.sum(a))
	print(np.min(a))
	print(np.max(a))
	pass


def array_basic2():
	a = np.arange(2,14).reshape((3,4))
	print(a)
	print(np.argmin(a))
	print(np.argmax(a))
	print(np.mean(a))
	print(np.mean(a,axis=0))
	print(np.average(a))
	print(np.median(a))
	print(np.cumsum(a))
	print(np.diff(a))
	print(np.nonzero(a))
	print(np.sort(a))
	print(np.transpose(a))
	print(a.T)
	print(np.clip(a,5,9))
	pass


def array_index():
	a = np.arange(3,15)
	print(a)
	print(a[3])
	a = np.arange(3,15).reshape((3,4))
	print(a[2][1])
	print(a[2,1])
	print(a[:,1])
	print(a[1,:])
	print(a[1,1:3])
	for row in a:
		print(row)
	for col in a.T:
		print(col)
	print(a.flatten())
	for item in a.flat:
		print(item)
	pass


def array_merge():
	a = np.array([1,1,1])
	b = np.array([2,2,2])
	print(np.vstack((a,b)))
	print(np.hstack((a,b)))
	print(a[:,np.newaxis])
	a = np.array([1,1,1])[:,np.newaxis]
	b = np.array([2,2,2])[:,np.newaxis]
	print(np.concatenate((a,b,b)))
	print(np.concatenate((a,b,b),axis=1))
	pass


def array_split():
	a = np.arange(12).reshape((3,4))
	print(a)
	print(np.split(a,2,axis=1))
	print(np.split(a,3,axis=0))
	print(np.array_split(a,2,axis=0))
	print(np.vsplit(a,3))
	print(np.hsplit(a,2))
	pass


def array_copy():
	a = np.arange(4)
	b = a
	c = a
	d = b
	e = a.copy()
	print(a)
	a[0] = 11
	print(a,b,d,e)
	b[1:3] = [22,33]
	print(a,b,d,e)
	pass


#array_attribute()
#array_create()
#array_basic()
#array_basic2()
#array_index()
#array_merge()
#array_split()
array_copy()




















