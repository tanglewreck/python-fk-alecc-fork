# coding: utf-8
"""perftest 2025-05-20"""

# pylint: disable=unused-import
from collections.abc import Callable
from functools import partial
import timeit
import numpy as np
from algorithms.bubblesort import bubblesort as bs
from algorithms.bubblesort import bubblesort2 as bs2
from algorithms.bubblesort import bubblesort3 as bs3
from algorithms.inssort import inssort as ins
from algorithms.inssort import inssort2 as ins2
from algorithms.inssort import inssort3 as ins3
from algorithms.inssort import inssortw_while as insww
from algorithms.inssort import inssortw_for as inswf
from algorithms.qsort import qsort as qs
from algorithms.qsort import qsort2 as qs2
from algorithms.qsort import qsorti as qsi
from algorithms.qsort import qsorti2 as qsi2


# Create some arrays
N = 2000
LS = np.arange(0, N + 1)  # Sorted array: [0...N]
LR = LS[::-1]  # Reverse-sorted array
LRAND = np.random.randint(0, 10_000, N)  # Random array
LRANDISH = np.append(np.random.randint(0, 2, int(N/2)),
                     np.arange(int(N/2) ))
LRANDISH = np.random.randint(0, 2, N)
LRANDISH2 = np.append(np.random.randint(0, 2, int(N/2)),
                      np.arange(int(N/2) )[::-1])
# Choose one to sort (below)
L = LR

# Algorithms to test
# pylint: disable=unused-variable
ALGS1 = [bs, bs2, bs3, ins, ins2, ins3, insww, qsi2]
ALGS2 = [bs, bs2, bs3, ins, ins2, ins3, inswf, insww, qsi, qsi2]
ALGS3 = [ins, ins2, ins3, inswf, insww, qsi, qsi2]
ALGS = ALGS3
# pylint: enable=unused-variable

# def fw(algo: Callable[[np.ndarray, int, int, bool, bool], tuple],
def fw(algo, l: np.ndarray) -> tuple:
    """wrapper"""
    return algo(l, 0, len(l) -1, True, False)[1:]


def tw(func, arr: np.ndarray, n: int):
    """tw"""
    f = partial(func, arr, 0, len(arr) -1, copylist=True)
    t = timeit.Timer(f)
    return t.timeit(n)


def main():
    """main"""
    print(f"N={N}")
    for algo in ALGS:
        print(f"{algo.__name__}")
        print("sorted:  \t", end="")
        t = tw(func=algo, arr=LS, n=1)
        print(f"{t:2.4f} s")
        #
        print("reversed:\t", end="")
        t = tw(func=algo, arr=LS, n=1)
        print(f"{t:2.4f} s")
        #
        print("random:  \t", end="")
        t = tw(func=algo, arr=LRAND, n=1)
        print(f"{t:2.4f} s")
        #
        print("randish:  \t", end="")
        t = tw(func=algo, arr=LRANDISH, n=1)
        print(f"{t:2.4f} s")
        #
        print("randish2:  \t", end="")
        t = tw(func=algo, arr=LRANDISH, n=1)
        print(f"{t:2.4f} s")
        # print()


if __name__ == "__main__":
    main()
