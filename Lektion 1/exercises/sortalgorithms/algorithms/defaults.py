"""
2025-02-19
Defaults for sort algorithms and
performance measurements.
"""
__all__ = ["ITERATIONS", "TIMEIT_ITERATIONS"]
__all__ += ["N", "MIN", "MAX"]

# Number of numbers in a list
N = 10
# Min size of a number
MIN = 1
# Max size of a number
MAX = 100

# Number of iterations while measuring performance
ITERATIONS = 100
# Number of iterations using the timeite module
TIMEIT_ITERATIONS = 100
# Number of measurements (the number of
# times to run each test)
MEASUREMENTS = 10
