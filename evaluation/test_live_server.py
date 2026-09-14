import unittest
from live_server import FramePairs

class PairTest(unittest.TestCase):
    def test_only_equal_frames_and_no_duplicates(self):
        p=FramePairs()
        self.assertIsNone(p.add(0,10,'a'))
        self.assertIsNone(p.add(1,11,'b'))
        self.assertEqual(p.add(1,10,'c'),('a','c'))
        self.assertIsNone(p.add(0,10,'a'))
        self.assertEqual(p.add(0,11,'d'),('d','b'))
    def test_reset_and_bounded_buffer(self):
        p=FramePairs()
        for i in range(100):p.add(0,i,i)
        self.assertLessEqual(len(p.pending),8)
        p.clear()
        self.assertIsNone(p.add(0,0,'a'))
        self.assertEqual(p.add(1,0,'b'),('a','b'))

if __name__=='__main__':unittest.main()
