import array as dataarray
import cStringIO

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

	def _offset_for_key(self, key):
		offset = 0
		for k_index, k in enumerate(key):
			offset += self.strides[k_index] * k
		return offset

	def _indices_for_slice_key(self, slice_key):
		indices = []
		for k_index, k in enumerate(slice_key):
			s = self.shape[k_index]
			indices.append(k.indices(s))
		return indices

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
				raise TypeError, "key elements must be instaces of int or slice."

		# Fill in missing axes with their full range.
		while len(slice_key) < len(self.shape):
			slice_key.append(slice(0, self.shape[len(slice_key)], 1))

		# Calculate new offset, shape & strides from the slices.
		indices = self._indices_for_slice_key(slice_key)

		offset = self._offset_for_key([i[0] for i in indices])
		shape = [_ceildiv(i[1] - i[0], i[2]) for i in indices]
		strides = [self.strides[i_index] * i[2] for i_index, i in enumerate(indices)]

		return ndarray(shape, self.data, offset, strides, self, self.typecode)

	def __setitem__(self, key, value):
		offset = self._offset_for_key(key)
		self.data[self.offset + offset] = value

	def __float__(self):
		return float(self.data[self.offset])

	def __int__(self):
		return int(self.data[self.offset])

	def __repr__(self):
		output = cStringIO.StringIO()

		def _repr_r(axis, offset):
			if axis < len(self.shape):
				output.write("\n")
				output.write("\t" * axis)
				output.write("[")

				for k_index, k in enumerate(xrange(self.shape[axis])):
					_repr_r(axis+1, offset + k * self.strides[axis])
					if k_index < self.shape[axis] - 1:
						output.write(", ")

				output.write("]")

			else:
				output.write("{0}".format(self.data[offset]))

		output.write("array(")
		_repr_r(0, self.offset)
		output.write(")")

		return output.getvalue()

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

def array(obj):
	s = _shape_from_object(obj)
	a = ndarray(s)
	_assign_from_object(a, obj)
	return a

def zeros(shape):
	return ndarray(shape)

print array(
	[
		[
			[
				1, 
				[1, 1, 1, 1], 
				[2, 2],
				[2, 2]
			], 
			1
		], 
		[2, 2, 1, 1], 
		[3, 3],
		[3, 3]
	])

test = ndarray((5,5,5))
print test
print test[0]
print test[0:4:2]
print test[1,2]
print int(test[1,2,3])

test = array([[1,2],[3,4]])
print test[0:2,1:]
print slice(None,None).indices(2)

test = array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print test[1::2,::3]
