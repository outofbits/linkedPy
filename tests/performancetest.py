import unittest
import time
import exec as lexec
import re
from os import listdir
from os.path import dirname, realpath, join, isfile, basename


class PerformanceTestMeta(type):
    """ This meta class is used to generate the test cases for given test files. """

    def __new__(mcs, name, bases, dct):
        test_resource_dir = join(dirname(realpath(__file__)), 'resources')

        def generate_test(test_file, test_result):
            def test(self):
                p_interpreter = 0.0
                linkedpy_interpreter = 0.0
                for run in range(self.test_runs):
                    # Determine timing of python interpreter
                    run_time = -time.time()
                    exec(compile(open(test_file, "rb").read(), test_file, 'exec'), {}, {})
                    run_time += time.time()
                    p_interpreter += run_time
                    # Determine timing of linkedpy interpreter
                    run_time = -time.time()
                    lexec.execute_program_file(test_file)
                    run_time += time.time()
                    linkedpy_interpreter += run_time
                self.assertGreater((p_interpreter / self.test_runs) * self.performance_goal,
                                   (linkedpy_interpreter / self.test_runs))

            return test

        # Reads in all test files and generates corresponding test cases.
        for file in [file for file in map(lambda x: join(test_resource_dir, x), listdir(test_resource_dir)) if
                     isfile(file) and file.endswith('.lpy')]:
            test_basename = re.sub(r'.lpy$', '', basename(file))
            test_result_file = join(test_resource_dir, '%s.0' % test_basename)
            if isfile(test_result_file):
                test_result = None
                with open(test_result_file, 'r') as tfp:
                    test_result = '\n'.join([line for line in tfp])
                dct['test_%s' % test_basename] = generate_test(file, test_result)
            else:
                raise ValueError('There is no result file for the performance test file %s' % test_basename)
        return type.__new__(mcs, name, bases, dct)


class PerformanceTest(unittest.TestCase, metaclass=PerformanceTestMeta):
    """ This class tests the performance of the interpreter in comparison with the given python interpreter."""

    performance_goal = 4000
    test_runs = 2


if __name__ == '__main__':
    unittest.main()
