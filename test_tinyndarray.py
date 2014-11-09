import tinyndarray
import numpy
import unittest

def _clean_repr(a):
	return "".join(repr(a).split())

class TestNDArray(unittest.TestCase):
	def setUp(self):
		self.t0 = tinyndarray.array([[1.0,2.0],[3.0,4.0]])
		self.n0 = numpy.array([[1.0,2.0],[3.0,4.0]])

	def test_repr(self):
		self.assertEqual(_clean_repr(self.t0), _clean_repr(self.n0))

	def test_slice(self):
		self.assertEqual(_clean_repr(self.t0[1:]), _clean_repr(self.n0[1:]))
		self.assertEqual(_clean_repr(self.t0[1:,1:]), _clean_repr(self.n0[1:,1:]))
		self.assertEqual(_clean_repr(self.t0[-1:,]), _clean_repr(self.n0[-1:,]))
		#self.assertEqual(_clean_repr(self.t0[-1:,1]), _clean_repr(self.n0[-1:,1]))

class TestNDIter(unittest.TestCase):
	def setUp(self):
		self.t0 = tinyndarray.array([[1,2],[3,4]])
		self.n0 = numpy.array([[1,2],[3,4]])

	def test_basic(self):
		pass
		#self.assertEqual(
		#	[i for i in tinyndarray.nditer(self.t0)],
		#	[i for i in numpy.nditer(self.n0)])

	def test_reversed(self):
		pass
		#self.assertEqual([i for i in reversed(tinyndarray.nditer(self.t0))], [i for i in reversed(numpy.nditer(self.n0))])

class TestFunctions(unittest.TestCase):
	def test_array(self):
		pass

	def test_zeros(self):
		pass

if __name__ == '__main__':
    unittest.main()