import unittest
from PulseCoding import generateSimplex, reconstructBGS
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_simplex(self):
        s = generateSimplex(2)[0]
        self.assertEqual(s[0].tolist(), [1, 0, 1])


if __name__ == '__main__':
    unittest.main()
