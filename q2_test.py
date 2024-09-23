import unittest
from KOERUIEN_32839677_ASSIGNMENT1 import TreeMap

class TestProblem2(unittest.TestCase):

    # staff provided - guaranteed correct
    def test_assignment_examples(self):
        # Example 1
        # The roads represented as a list of tuples
        roads = [(0,1,4), (1,2,2), (2,3,3), (3,4,1), (1,5,2),
        (5,6,5), (6,3,2), (6,4,3), (1,7,4), (7,8,2),
        (8,7,2), (7,3,2), (8,0,11), (4,3,1), (4,8,10)]
        # The solulus represented as a list of tuples
        solulus = [(5,10,0), (6,1,6), (7,5,7), (0,5,2), (8,4,8)]
        # Creating a TreeMap object based on the given roads
        myforest = TreeMap(roads, solulus)

        # Example 1.1
        start = 1
        exits = [7, 2, 4]

        got = myforest.escape(start, exits)
        expected = (9, [1, 7])
        self.assertEqual(expected, got)

        # Example 1.2
        start = 7
        exits = [8]

        got = myforest.escape(start, exits)
        expected = (6, [7, 8])
        self.assertEqual(expected, got)

        # Example 1.3
        start = 1
        exits = [3, 4]
        got = myforest.escape(start, exits)
        expected = (10, [1, 5, 6, 3])
        self.assertEqual(expected, got)

        # Example 1.4
        start = 1
        exits = [0, 4]
        got = myforest.escape(start, exits)
        expected = (11, [1, 5, 6, 4])
        self.assertEqual(expected, got)

        # Example 1.5
        start = 3
        exits = [4]
        got = myforest.escape(start, exits)
        expected = (20, [3, 4, 8, 7, 3, 4])
        self.assertEqual(expected, got)

        # Example 1.6
        start = 8
        exits = [2]
        got = myforest.escape(start, exits)
        expected = (16, [8, 0, 2])
        self.assertEqual(expected, got)
    
    # student provided
    def test_example_1(self): # Nam Pham
        roads = [ 
            (0,1,1),
            (1,2,1),
            (2,3,1),
            (2,5,100),
            (3,4,1),
            (3,5,1),
            (4,2,1),
            (4,3,1),
        ]

        # The solulus represented as a list of tuples
        solulus = [(3, 1, 4)]

        # Creating a TreeMap object based on the given roads
        myforest = TreeMap(roads, solulus)

        self.assertEqual(myforest.escape(1, [5]), (5, [1, 2, 3, 4, 3, 5]))
    
    def test_example_2(self): # Michael Wang
        forest3 = TreeMap([(0,1,1), (4, 3, 3), (3, 2, 3), (1, 2, 3)], [(3, 5, 2)])
        self.assertIsNone(forest3.escape(4, [1]))
    
    def test_example_3(self): # Me
        myforest = TreeMap([(0,1,1), (1,2,1), (2,3,1), (3,4,1), (4,5,1)], [(1,1,5)])
        for i in range(1, 5):
            self.assertIsNone(myforest.escape(1, [i]))
        self.assertEqual(myforest.escape(1, [5]), (1, [1,5]))
    
    def test_example_4(self): # Me
        myforest = TreeMap([(0,1,1), (1,2,1), (2,3,1), (3,4,1), (4,5,1)], [(1,5,5), (2,1,5)])
        self.assertEqual(myforest.escape(1, [5]), (2, [1,2,5]))
    
    def test_example_5(self): # Me
        myforest = TreeMap([(0,1,1), (1,2,1), (2,3,1), (3,4,1), (4,5,1)], [(0,1,3), (3,1,5)])
        self.assertEqual(myforest.escape(0, [5]), (3, [0,3,4,5]))

    def test_example_6(self): # Me
        myforest = TreeMap([(0,1,1), (1,2,1), (2,3,1)], [(0,1,1)])
        self.assertEqual(myforest.escape(0,[3]), (3,[0,1,2,3]))
    
    def test_example_7(self): # Me
        myforest = TreeMap([(0,1,1), (1,2,1)], [(0,100,1)])
        self.assertEqual(myforest.escape(0, [2]), (101, [0,1,2]))
    
    def test_example_8(self): # Me
        myforest = TreeMap([(0,1,1)], [(1,100,0)])
        self.assertEqual(myforest.escape(0, [0]), (101, [0,1,0]))

    def test_long_path(self): # Jiayi
        myforest = TreeMap([(0,1,1), (1, 2, 1), (2, 3, 1), (3, 4, 1)], [(4,4,0)])
        self.assertEqual(myforest.escape(0, [4]), (12, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))
                         
if __name__ == '__main__':
    unittest.main()   