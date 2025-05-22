#!/usr/bin/env python3
# coding: utf-8
"""
    NAME
        measure
    DESCRIPTION
        Measure performance
    AUTHOR
        mier
    DATE
        2025-05-07
"""
__all__ = ["genarr",
           "measure",
           "measurements",
           "main_function"]

# pylint: disable=unused-import
import timeit

from collections.abc import Callable
from functools import partial
# from typing import

import numpy as np
import pandas as pd
from pandas import DataFrame

from algorithms.utils import debug_msg, err_msg, sys
from algorithms.defaults import ALGORITHMS, ALGOSALL
from algorithms.bubblesort import bubblesort
from algorithms.bubblesort import bubblesort2
from algorithms.bubblesort import bubblesort3
from algorithms.inssort import inssort
from algorithms.inssort import inssort2
from algorithms.inssort import inssort3
from algorithms.inssort import inssortw_for
from algorithms.inssort import inssortw_while
from algorithms.defaults import qsort
from algorithms.defaults import qsort2
from algorithms.defaults import qsorti
from algorithms.defaults import qsorti2
# pylint: enable=unused-import
from algorithms.defaults import LENGTH_DEFAULT
from algorithms.defaults import LIST_LENGTHS
from algorithms.defaults import LOWER, UPPER
from algorithms.defaults import ITERATIONS


def genarr(low: int = LOWER, high: int = UPPER,
           size=(ITERATIONS, LENGTH_DEFAULT),
           dtype=int):
    """
        Basically a wrapper around np.random.randint with
        default values provided.

        Parameters
        ----------
            low : int
                  Smallest number to generate
            high : int
                   Largest number to generate
            size : tuple of ints
                   Number of lists and number of numbers
                   in each list, default: (ITERATIONS, LENGTH_DEFAULT)
            dtype : defaults to int
        Returns
        -------
            np.ndarray of random numbers (of type 'dtype') in the
            half-open interval [low, high) of size 'size'.
    """
    return np.random.randint(low, high, size, dtype)


# pylint: disable=too-many-locals
def measure(ldata: np.ndarray,
            algo: Callable,
            # algo: Callable[[np.ndarray, int, int, bool, bool], tuple],
            nlists: int = ITERATIONS, llength: int = 10,
            verbose=0):
    """Measure performance of a sorting algorithm."""
    t, comps, swaps = [], [], []
    try:
        # Get number of ldata to sort (nlists)
        # and length of each list (llength) from
        # the shape (2 x 2 tuple) of the ldata
        # variable (np.array):
        # nlists, llength = ldata.shape
        # nlists, llength = ITERATIONS
        if verbose > 0:
            print(f"algorithm {algo.__name__}")
            print(f"ldata.shape {ldata.shape}, "
                  f"llength {llength}, "
                  f"nlists {nlists}")
        if llength > ldata.shape[1]:
            errormsg = f"llength ({llength}) > " + \
                    "(list-length in ldata ({ldata.shape[1]}))"
            raise IndexError(errormsg)
        for k in range(nlists):
            lslice = ldata[k][:llength]
            if verbose > 2:
                print(lslice)
            # Measure number of comparisons and swaps
            _, c, s = algo(lslice, 0, len(lslice) - 1)
            comps.append(c)
            swaps.append(s)
            # Measure execution time
            f = partial(algo, lslice, 0, len(lslice) - 1)
            timer = timeit.Timer(f)
            t.append(timer.timeit(1))
        if verbose > 1:
            print(f"t = {t}")
            print(f"comps = {comps}")
            print(f"swaps = {swaps}")
            print()
        if verbose > 0:
            print(f"t.mean = {np.mean(t) / 1e-3:01.2f} ± "
                  f"{np.std(t) / 1e-3:01.2f} ms")
            print(f"t.min = {np.min(t) / 1e-3:01.2f} ms")
            print(f"t.max = {np.max(t) / 1e-3:01.2f} ms")
            print()
        if verbose >= 1.5:
            print(f"comps.mean = {np.mean(comps):01.2f} ± "
                  f"{np.std(comps):01.2f}")
            print(f"swaps.mean = {np.mean(swaps):01.2f} ± "
                  f"{np.std(swaps):01.2f}")
            print()
    except AttributeError as exception:
        err_msg("Got an AttributeError:")
        err_msg(f"{repr(exception)}")
        raise
    except IndexError as exception:
        err_msg("Got an IndexError:")
        err_msg(f"{repr(exception)}")
        raise
    # return np.mean(t), np.mean(comps), np.mean(swaps)
    # return np.array(t), np.array(comps), np.array(swaps)
    return DataFrame([t, comps, swaps],
                     index=["t", "comps", "swaps"])


def measurements(ldata: np.ndarray,
                 algo: Callable[[np.ndarray, int, int, bool, bool], tuple],
                 llengths=LIST_LENGTHS, nlists=(ITERATIONS,),
                 verbose=False) -> None:
    """
        Measure performance of a sort algorithm.
        Wrapper around measure().

        Parameters
        ----------
        ldata : list or numpy array. If None, a (rather large) two-dimensional
                array is generated using genarr().
        algo : Callable – the sort algorithm (not a function *name*).
               Defaults to bubblesort().
        llengths : iterable – a list of listlengths to be tested. Defaults
               to LIST_LENGTHS, imported from algorithms.default.
        nlists : list – the number of lists to be tested. Defaults to
                 ITERATIONS, imported from algorithms.defaults.

    """
    if max(nlists) > len(ldata):
        errormsg = f"max(nlists) ({max(nlists)}) is too large " \
                   f"(ldata contains only {len(ldata)} lists)"
        raise IndexError(errormsg)
    try:
        print(f"Measuring performance of {algo.__name__}")
        if ldata is None:
            ldata = genarr()
            llengths = [LENGTH_DEFAULT, ]
        for ll in llengths:
            print("_________________")
            print(f"list-length: {ll}",)
            print("_________________")
            for nl in nlists:
                df = measure(ldata, algo, nlists=nl,
                             llength=ll, verbose=verbose)
                print(f"    nlists (iterations): {nl} :", end="")
                print(f'{"t"}: {df.loc["t"].mean() / 1e-3:2.4f} ± '
                      f'{df.loc["t"].std() / 1e-3:2.4f} ms', end=" ")
                for ind in df.index[1:]:
                    print(f'{ind}: {df.loc[ind].mean():2.2f} ± '
                          f'{df.loc[ind].std():2.2f}', end=" ")
                print()
        print()
    except AttributeError as exception:
        err_msg("Got an AttributeError:")
        err_msg(f"{repr(exception)}")
        return None
    except IndexError as exception:
        err_msg("Got an IndexError:")
        err_msg(f"{repr(exception)}")
    return None


def main_function() -> None:
    """main"""
    # llmax = 50_000
    # llmax = 40_000
    # for ll in [10_000, 15_000,
    #            20_000, 30_000,
    #            40_000]:
    # for ll in [100, 250, 500, 750, 1000]:
    nlists = 5
    ll_lower = 1000
    ll_upper = 10000
    ll_step = 500
    ll_range = range(ll_lower, ll_upper + 1, ll_step)
    data = genarr(size=(nlists, ll_upper))
    for ll in ll_range:
        print("-" * 30)
        # measure(ldata=data, algo=bubblesort,
        #         nlists=nlists, llength=ll, verbose=1)
#         measure(ldata=data, algo=inssort3,
#                 nlists=nlists, llength=ll, verbose=1)
#         measure(ldata=data, algo=qsort2,
#                 nlists=nlists, llength=ll, verbose=1)
        # data = np.random.randint(LOWER, UPPER, (nlists, ll), dtype=int)
        algos_list = [
                # bubblesort, bubblesort2, bubblesort3,
                # inssort, inssort2, inssort3,
                # inssortw_for,
                inssortw_while,
                # qsort, qsort2,
                # qsorti,
                qsorti2]
        for algorithm in algos_list:
            measure(ldata=data, algo=algorithm,  # type: ignore[arg-type]
                    nlists=nlists, llength=ll, verbose=1)
        #measure(ldata=data, algo=qsorti,
        #        nlists=nlists, llength=ll, verbose=1)
        #measure(ldata=data, algo=qsorti2,
        #        nlists=nlists, llength=ll, verbose=1)
        # nlists=np.arange(30, 60, 10))
    print()
    print("-" * 30)
#    measurements(ldata=data,
#                 algo=inssort3,
#                 llengths=np.arange(100, 1001, 100),
#                 nlists=(30, ))
    # print(df)


if __name__ == "__main__":
    main_function()
