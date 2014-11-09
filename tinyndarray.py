import array as dataarray
import io

def _ceildiv(a, b):
	return -(-a // b)

def _strides_for_shape(shape):
	strides = []
	stride_product = 1
	for d in shape:
		strides.append(stride_product)
		stride_product *= d
	strides.reverse()
	return strides

def _size_for_shape(shape):
	stride_product = 1
	for d in shape:
		stride_product *= d
	return stride_product

def _shape_from_object(obj):
	shape = []

	def _shape_from_object_r(index, element, depth):
		try:
			for i, e in enumerate(element):
				_shape_from_object_r(i, e, depth+1)
			while len(shape) <= depth:
				shape.append(0)
			shape[depth] = max(shape[depth], i+1)
		except TypeError:
			pass

	_shape_from_object_r(0, obj, 0)

	return shape

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
	for depth in reversed(xrange(len(shape))):
		key[depth] += 1
		if key[depth] < shape[depth]:
			return True
		if depth == 0:
			return False
		key[depth] = 0

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
		# http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#arrays-indexing

		# Promote to tuple.
		if not isinstance(key, tuple):
			key = (key,)

		# Convert the key to slices and return a new array view.
		slice_key = []
		for k in key:
			if isinstance(k, slice):
				slice_key.append(k)
			elif isinstance(k, int):
				slice_key.append(slice(k, k+1, 1))
			else:
				raise(TypeError, "key elements must be instaces of int or slice.")

		# Fill in missing axes with their full range.
		while len(slice_key) < len(self.shape):
			slice_key.append(slice(0, self.shape[len(slice_key)], 1))

		# Calculate new offset, shape & strides from the slices.
		indices = _slice_indices_for_slice_key(slice_key, self.shape)

		offset = _offset_for_key([i[0] for i in indices], self.strides)
		shape = [_ceildiv(i[1] - i[0], i[2]) for i in indices]
		strides = [self.strides[i_index] * i[2] for i_index, i in enumerate(indices)]

		return ndarray(shape, self.data, offset, strides, self, self.typecode)

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
				s += repr(self.data[offset]).rstrip('0')
				#s += repr(self.data[offset])
			return s

		s = 'array('
		s = _repr_r(s, 0, self.offset)
		s += ')'

		return s

def array(obj):
	s = _shape_from_object(obj)
	a = ndarray(s)
	_assign_from_object(a, obj)
	return a

def zeros(shape):
	return ndarray(shape)
