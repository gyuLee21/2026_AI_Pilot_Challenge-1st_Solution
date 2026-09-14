import unittest
from concurrent.futures import Future
from parallel_tournament import PrefetchedPairs, equivalent, choose_workers


class FakeExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, fn, task):
        self.tasks.append(task)
        future = Future()
        if task[0] == -1:
            future.set_exception(RuntimeError('evaluation failed'))
        else:
            future.set_result(task)
        return future


class ParallelTest(unittest.TestCase):
    def test_selection_requires_accuracy_and_useful_speedup(self):
        baseline = dict(workers=1, seconds=100, exact_match=True)
        bad = dict(workers=4, seconds=10, exact_match=False)
        small_gain = dict(workers=2, seconds=95, exact_match=True)
        self.assertEqual(choose_workers([baseline, bad, small_gain]), 1)
        self.assertEqual(choose_workers([baseline, bad, dict(small_gain, seconds=80)]), 2)

    def test_bounded_unique_dispatch(self):
        executor = FakeExecutor()
        tasks = [(i, i+1, 42) for i in range(10)]
        evaluate = PrefetchedPairs(executor, tasks, 2)
        self.assertEqual(len(executor.tasks), 2)
        for task in tasks:
            self.assertEqual(evaluate(*task), task)
            self.assertLessEqual(len(evaluate.pending), 2)
        self.assertEqual(executor.tasks, tasks)

    def test_failure_propagates_without_scheduling_more(self):
        executor = FakeExecutor()
        evaluate = PrefetchedPairs(executor, [(-1, 1, 42), (2, 3, 42)], 1)
        with self.assertRaises(RuntimeError):
            evaluate(-1, 1, 42)
        self.assertEqual(len(executor.tasks), 1)

    def test_compare_all_outcomes_but_not_wall_clock(self):
        result = dict(records=[dict(score=1, crash=False)], summary=dict(score=1), wall_sec=1)
        self.assertTrue(equivalent(result, dict(result, wall_sec=2)))
        self.assertFalse(equivalent(result, dict(result, records=[dict(score=1, crash=True)])))


if __name__ == '__main__':
    unittest.main()
