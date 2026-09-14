import unittest
from scripts import damage_fork


class PulseTests(unittest.TestCase):
    def test_exact_window_and_resume(self):
        pulse = {'start_additional': 547, 'end_additional': 646, 'lr': 6e-5}
        lr = getattr(damage_fork, 'main_lr', lambda i, p: 5e-5)
        self.assertEqual(lr(546, pulse), 5e-5)
        self.assertEqual(lr(547, pulse), 6e-5)
        self.assertEqual(lr(646, pulse), 6e-5)
        self.assertEqual(lr(647, pulse), 5e-5)
        self.assertEqual(sum(lr(i, pulse) == 6e-5 for i in range(1, 4501)), 100)
        self.assertEqual(sum(lr(i, pulse) == 6e-5 for i in range(601, 4501)), 46)
        self.assertEqual(lr(600, None), 5e-5)
