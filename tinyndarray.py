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
			shape[axis] = max(shape[axis], i+1)
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
			array[key] = element

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

def _slice_indices_for_slice_key(slice_key, shape):
	indices = []
	for k_index, k in enumerate(slice_key):
		s = shape[k_index]
		indices.append(k.indices(s))
	return indices

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
	def __init__(self, shape, buffer=None, offset=0, strides=None, base=None, typecode='f'):
		self.shape = shape
		self.offset = offset

		if strides:
			self.strides = strides
		else:
			self.strides = _strides_for_shape(shape)

		self.ndim = len(shape)
		self.size = self.strides[0] * self.shape[0]

		self.base = base
		self.typecode = typecode

		if buffer:
			self.data = buffer
		else:
			self.data = dataarray.array(typecode, [0] * self.size)

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

		return ndarray(tuple(shape), self.data, offset, tuple(strides), self, self.typecode)

	def __setitem__(self, key, value):
		offset = _offset_for_key(key, self.strides)
		self.data[self.offset + offset] = value

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

newaxis = None

def array(obj):
	s = _shape_from_object(obj)
	a = ndarray(s)
	_assign_from_object(a, obj)
	return a

def zeros(shape):
	return ndarray(shape)
