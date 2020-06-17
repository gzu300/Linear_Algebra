import unittest
from pkg import Linear_Algebra
import numpy as np

class TestLU(unittest.TestCase):
    def setUp(self):
        self.U_answer = np.around(np.array([[2,1,0],[0,3/2,1],[0,0,4/3]], dtype=float), decimals=2).tolist()
        self.L_answer = np.around(np.array([[1,0,0],[1/2,1,0],[0,2/3,1]], dtype=float), decimals=2).tolist()

    def test_perm(self):
        answer = np.array([[0,1,0], [1,0,0], [0,0,1]], dtype=float).tolist()
        result = Linear_Algebra.make_perm_mx(3, 0, 1).tolist()
        self.assertEqual(result, answer)

    def test_LU(self):
        L_result, U_result = np.around(Linear_Algebra.LU(np.array([[2,1,0],[1,2,1],[0,1,2]], dtype=float)), decimals=2).tolist()
        self.assertEqual(U_result, self.U_answer)
        self.assertEqual(L_result, self.L_answer)
    
class TestDet(unittest.TestCase):
    def setUp(self):
        self.input_mx = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], dtype=float)
    def test_find_det(self):
        result = np.around(Linear_Algebra.find_det(A = self.input_mx), decimals=2).tolist()
        answer = np.around(5, decimals=2).tolist()
        self.assertEqual(result, answer)

if __name__ == '__main__':
    unittest.main()