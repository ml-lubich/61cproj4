from utils import *
from unittest import TestCase

"""
For each operation, you should write tests to test  on matrices of different sizes.
Hint: use dp_mc_matrix to generate dumbpy and numc matrices with the same data and use
      cmp_dp_nc_matrix to compare the results
"""
class TestSanity(TestCase):
    def test_slices(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(20, 20, seed=346)
        dp_slice = dp_mat[-23:25,2]
        nc_slice = nc_mat[-23:25,2]
        self.assertTrue(cmp_dp_nc_matrix(dp_slice, nc_slice))
class TestAdd(TestCase):
    def test_small_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(100, 250, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(250, 100, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(10000, 10000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(10000, 10000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestSub(TestCase):
    def test_small_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(100, 100, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(100, 100, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(10000, 10000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(10000, 10000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestAbs(TestCase):
    def test_small_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(100, 100, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(10000, 10000, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestNeg(TestCase):
    def test_small_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)
    def test_medium_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(100, 100, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(10000,10000, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestMul(TestCase):
    def test_small_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_petite_mul(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(25, 25, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(25, 25, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_average_mul(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(50, 50, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(50, 50, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(100, 250, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(250, 100, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_square_mul(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(100, 100, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(100, 100, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_mul(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(1000, 2500, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2500, 1500, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestPow(TestCase):
    def test_small_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        # print("\nINPUT:")
        # print(f"dp: {dp_mat}")
        # print(f"nc: {nc_mat}")
        power = 5
        is_correct, speed_up = compute([dp_mat, power], [nc_mat, power], "pow", False)
        self.assertTrue(is_correct)
        print_speedup(speed_up)
        # test identity
        is_correct, speed_up = compute([dp_mat, 0], [nc_mat, 0], "pow", False)
        self.assertTrue(is_correct)

    def test_medium_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(100, 100, seed=0)
        is_correct, speed_up = compute([dp_mat, 37], [nc_mat, 37], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(500, 500, seed=0)
        is_correct, speed_up = compute([dp_mat, 40], [nc_mat, 40], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestGet(TestCase):
    def test_get(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))

class TestSet(TestCase):
    def test_set(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEquals(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))

class TestShape(TestCase):
    def test_shape(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(dp_mat.shape == nc_mat.shape)
