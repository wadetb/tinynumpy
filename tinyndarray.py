import array as dataarray
import io

def _ceildiv(a, b):
	return -(-a // b)

def _strides_for_shape(shape):
	strides = []
	stride_product = 1
	for s in shape:
		strides.append(stride_product)
		stride_product *= s
	strides.reverse()
	return tuple(strides)

def _size_for_shape(shape):
	stride_product = 1
	for s in shape:
		stride_product *= s
	return stride_product

def _shape_from_object(obj):
	shape = []

	def _shape_from_object_r(index, element, axis):
		try:
			for i, e in enumerate(element):
				_shape_from_object_r(i, e, axis+1)
			while len(shape) <= axis:
				shape.append(0)
			l = i + 1
			s = shape[axis]
			if l > s:
				shape[axis] = l
		except TypeError:
			pass

	_shape_from_object_r(0, obj, 0)
	return tuple(shape)

def _assign_from_object(array, obj):
	key = []

	def _assign_from_object_r(element):
		try:
			for i, e in enumerate(element):
				key.append(i)
				_assign_from_object_r(e)
				key.pop()
		except TypeError:
			array[tuple(key)] = element

	_assign_from_object_r(obj)

def _increment_mutable_key(key, shape):
	for axis in reversed(xrange(len(shape))):
		key[axis] += 1
		if key[axis] < shape[axis]:
			return True
		if axis == 0:
			return False
		key[axis] = 0

def _key_for_index(index, shape):
	key = []
	for s in shape[1:]:
		n = index / s
		key.append(n)
		index -= n * s
	key.append(index)
	return tuple(key)

def _offset_for_key(key, strides):
	offset = 0
	for k_index, k in enumerate(key):
		offset += strides[k_index] * k
	return offset

class nditer:
	def __init__(self, array):
		self.array = array
		self.key = [0] * len(self.array.shape)

	def __iter__(self):
		return self

	def __len__(self):
		return _size_for_shape(self.array.shape)

	def __getitem__(self, index):
		key = _key_for_index(index, self.array.shape)
		return self.array[key]

	def __setitem__(self, index, value):
		key = _key_for_index(index, self.array.shape)
		self.array[key] = value

	def __next__(self):
		if self.key is None:
			raise StopIteration
		value = self.array[tuple(self.key)]
		if not _increment_mutable_key(self.key, self.array.shape):
			self.key = None
		return value

	def next(self):
		return self.__next__()

class ndarray:
	def __init__(self, shape, offset=0, strides=None, typecode='f', base=None):
		self.shape = shape
		self.offset = offset

		if strides:
			self.strides = strides
		else:
			self.strides = _strides_for_shape(shape)

		self.ndim = len(shape)
		self.size = _size_for_shape(shape)

		self.typecode = typecode

		self.base = base
		if base:
			self.data = base.data
		else:
			self.data = dataarray.array(typecode, [0] * self.size)

	def __len__(self):
		return self.size

	def __getitem__(self, key):
		# Indexing spec is located at:
		# http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

		# Promote to tuple.
		if not isinstance(key, tuple):
			key = (key,)

		axis = 0
		shape = []
		strides = []
		offset = self.offset

		for k in key:
			if isinstance(k, int):
				offset += k * self.strides[axis]
				axis += 1
			elif isinstance(k, slice):
				start, stop, step = k.indices(self.shape[axis])
				shape.append(_ceildiv(stop - start, step))
				strides.append(step * self.strides[axis])
				offset += start * self.strides[axis]
				axis += 1
			elif k is Ellipsis:
				raise(TypeError, "ellipsis are not supported.")
			elif k is None:
				shape.append(1)
				strides.append(0)
			else:
				raise(TypeError, "key elements must be instaces of int or slice.")

		shape.extend(self.shape[axis:])
		strides.extend(self.strides[axis:])

		if len(shape) == 0:
			shape = [1]
			strides = [0]

		return ndarray(tuple(shape), offset, tuple(strides), self.typecode, self)

	def __setitem__(self, key, value):
		view = self[key]
		mutable_key = [0] * len(view.shape)
		while True:
			offset = view.offset + _offset_for_key(mutable_key, view.strides)
			view.data[offset] = float(value)
			if not _increment_mutable_key(mutable_key, view.shape):
				break

	def __float__(self):
		return float(self.data[self.offset])

	def __int__(self):
		return int(self.data[self.offset])

	def __repr__(self):
		def _repr_r(s, axis, offset):
			if axis < len(self.shape):
				s += '\n' + ('\t' * axis) + '['
				for k_index, k in enumerate(xrange(self.shape[axis])):
					s = _repr_r(s, axis+1, offset + k * self.strides[axis])
					if k_index < self.shape[axis] - 1:
						s += ', '
				s += ']'
			else:
				s += repr(self.data[offset])
			return s

		s = 'array('
		s = _repr_r(s, 0, self.offset)
		s += ')'

		return s

	def all(self, axis=None):
		return all(self, axis)

	def any(self, axis=None):
		return any(self, axis)

	def argmax(self, axis=None):
		return argmax(self, axis)

	def argmin(self, axis=None):
		return argmin(self, axis)

	def copy(self):
		a = empty(self.shape)
		self_iter = nditer(self)
		a_iter = nditer(a)
		for i in xrange(self.size):
			a_iter[i] = self_iter[i]
		return a

	def clip(self, a_min, a_max, out=None):
		return clip(self, a_min, a_max, out)

	def cumprod(self, axis=None, out=None):
		return cumprod(self, axis, out)

	def cumsum(self, axis=None, out=None):
		return cumsum(self, axis, out)

	def fill(self, value):
		fill(self, value)

	def flatten(self):
		a = empty((self.size,))
		for i_index, i in enumerate(nditer(self)):
			a[i_index,] = float(i)
		return a

	def max(self, axis=None):
		return max(self, axis)

	def mean(self, axis=None):
		return mean(self, axis)

	def min(self, axis=None):
		return min(self, axis)

	def prod(self, axis=None):
		return prod(self, axis)

	def ptp(self, axis=None):
		return ptp(self, axis)

	def ravel(self):
		return ravel(self)

	def repeat(self, repeats, axis=None):
		return repeat(self, repeats, axis)

	def reshape(self, newshape):
		return reshape(self, newshape)

	def sum(self, axis=None):
		return sum(self, axis)

def all(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		for i in nditer(a):
			if float(i) == 0:
				return False
		return True

def any(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		for i in nditer(a):
			if float(i) != 0:
				return False
		return True

def argmax(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		r = float(nditer(a)[0])
		r_index = 0
		for i_index, i in enumerate(nditer(a)):
			v = float(i)
			if v > r:
				r = v
				r_index = i_index
		return r_index

def argmin(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		r = float(nditer(a)[0])
		r_index = 0
		for i_index, i in enumerate(nditer(a)):
			v = float(i)
			if v < r:
				r = v
				r_index = i_index
		return r_index

def clip(a, a_min, a_max, out):
	if out == None:
		out = empty(self.shape)
	a_iter = nditer(a)
	out_iter = nditer(out)
	for i in xrange(self.size):
		v = a_iter[i]
		if v > a_max:
			out_iter[i] = a_max
		elif v < a_min:
			out_iter[i] = a_min
		else:
			out_iter[i] = v
	return out

def cumprod(a, axis=None, out=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		if out == None:
			out = empty((self.size,))
		p = 1.0
		a_iter = nditer(a)
		out_iter = nditer(out)
		for i in xrange(self.size):
			p *= a_iter[i]
			out_iter[i] = p
		return out

def cumsum(a, axis=None, out=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		if out == None:
			out = empty((self.size,))
		s = 0.0
		a_iter = nditer(a)
		out_iter = nditer(out)
		for i in xrange(self.size):
			s += a_iter[i]
			out_iter[i] = s
		return out

def fill(a, value):
	for i in nditer(a):
		i[0,] = value

def max(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		r = float(nditer(a)[0])
		for i in nditer(a):
			v = float(i)
			if v > r:
				r = v
		return r

def mean(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		s = 0.0
		for i in nditer(a):
			s += float(i)
		return s / float(a.size)

def min(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		r = float(nditer(a)[0])
		for i in nditer(a):
			v = float(i)
			if v < r:
				r = v
		return r

def prod(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		p = 1.0
		for i in nditer(a):
			p *= float(i)
		return p

def ptp(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		mn = float(nditer(a)[0])
		mx = float(nditer(a)[0])
		p = 1.0
		for i in nditer(a):
			v = float(i)
			if v > mx:
				mx = v
			if v < mn:
				mn = v
		return mx - mn

def ravel(a):
	out = empty((a.size,))
	for i_index, i in enumerate(nditer(a)):
		out[i_index,] = float(i)
	return out

def repeat(a, repeats, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		out = empty((a.size * repeats,))
		for i_index, i in enumerate(nditer(a)):
			for r_index in xrange(repeats):
				out[i_index * repeats + r_index,] = float(i)
		return out

def reshape(a, newshape):
	out = a.copy()
	out.shape = newshape
	out.strides = _strides_for_shape(newshape)
	out.size = _size_for_shape(newshape)
	out.ndim = len(newshape)
	return out

def sum(a, axis=None):
	if axis:
		raise (TypeError, "axis argument is not supported")
	else:
		s = 0.0
		for i in nditer(a):
			s += float(i)
		return s

newaxis = None

def arange(size):
	a = empty((size,))
	a_iter = nditer(a)
	for i in xrange(size):
		a_iter[i] = i
	return a

def array(obj):
	shape = _shape_from_object(obj)
	a = ndarray(shape)
	_assign_from_object(a, obj)
	return a

def eye(size):
	a = zeros((size,size))
	for i in xrange(size):
		a[i,i] = 1.0
	return a

def zeros(shape):
	return ndarray(shape)

def empty(shape):
	return ndarray(shape)
