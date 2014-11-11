import tinyndarray
import numpy
import unittest

numpy.set_printoptions(formatter={'float': repr})

def _clean_repr(a):
	return "".join(repr(a).split())

class TestNDArray(unittest.TestCase):
	def setUp(self):
		a = [[ 1.0, 2.0, 3.0, 4.0],
		     [ 5.0, 6.0, 7.0, 8.0],
		     [ 9.0,10.0,11.0,12.0],
		     [13.0,14.0,15.0,16.0]]
		self.t0 = tinyndarray.array(a)
		self.n0 = numpy.array(a)

	def test_repr(self):
		self.assertEqual(_clean_repr(self.t0), _clean_repr(self.n0))

	def test_index(self):
		self.assertEqual(float(self.t0[1,1]), float(self.n0[1,1]))

	def test_slice(self):
		self.assertEqual(_clean_repr(self.t0[1:]), _clean_repr(self.n0[1:]))
		self.assertEqual(_clean_repr(self.t0[1:,1:]), _clean_repr(self.n0[1:,1:]))
		self.assertEqual(_clean_repr(self.t0[-1:,]), _clean_repr(self.n0[-1:,]))
		self.assertEqual(_clean_repr(self.t0[-1:,1]), _clean_repr(self.n0[-1:,1]))

	def test_double_slice(self):
		self.assertEqual(_clean_repr(self.t0[1:][2::2]), _clean_repr(self.n0[1:][2::2]))

	def test_newaxis(self):
		self.assertEqual(self.t0[tinyndarray.newaxis,2:].shape, (1,2,4))
		self.assertEqual(self.n0[numpy.newaxis,2:].shape, (1,2,4))
		self.assertEqual(_clean_repr(self.t0[tinyndarray.newaxis,2:]), _clean_repr(self.n0[numpy.newaxis,2:]))

class TestNDIter(unittest.TestCase):
	def setUp(self):
		self.t0 = tinyndarray.array([[1.0,2.0],[3.0,4.0]])
		self.n0 = numpy.array([[1.0,2.0],[3.0,4.0]])

	def test_basic(self):
		self.assertEqual(
			[float(i) for i in tinyndarray.nditer(self.t0)],
			[float(i) for i in numpy.nditer(self.n0)])

	def test_reversed(self):
		# NumPy reversed just returns array(1.0).
		self.assertEqual(
			[float(i) for i in reversed(tinyndarray.nditer(self.t0))], 
			[4.0, 3.0, 2.0, 1.0])

class TestFunctions(unittest.TestCase):
	def setUp(self):
		a0 = [
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
			]
		self.t0 = tinyndarray.array(a0)

		a1 = [[1,2],[3,4]]
		self.t1 = tinyndarray.array(a1)

	def test_array(self):
		# NumPy requires that the input array be full in all dimensions, so don't check compatibility.
		self.assertEqual(self.t0.shape, (4, 4, 4, 4))

	def test_zeros(self):
		pass

	def test_sum(self):
		self.assertEqual(self.t1.sum(), 10)

	def test_max(self):
		self.assertEqual(self.t1.max(), 4)

	def test_min(self):
		self.assertEqual(self.t1.min(), 1)

	def test_fill(self):
		t = tinyndarray.zeros((4, 4, 4))
		t.fill(1)
		self.assertEqual(t.sum(), 4*4*4)

	def test_copy(self):
		t = self.t1.copy()
		t[0,0] = 0
		self.assertEqual(self.t1.sum(), 10)
		self.assertEqual(t.sum(), 9)

	def test_flatten(self):
		self.assertEqual(_clean_repr(self.t1.flatten()), _clean_repr(tinyndarray.array([1, 2, 3, 4])))

if __name__ == '__main__':
    unittest.main()