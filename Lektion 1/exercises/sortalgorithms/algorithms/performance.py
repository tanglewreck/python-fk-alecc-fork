"""
    NAME
        performance.py
    DESCRIPTION
        Defines functions used to measure performance
        of sort algorithms.
    DATE
        2025-05-04
"""

import timeit
from collections.abc import Callable
from functools import partial
import numpy as np
from pandas import DataFrame
from . defaults import ITERATIONS
from . defaults import LENGTH_DEFAULT
from . defaults import LOWER, UPPER
# from . defaults import MAXLENGTH
# pylint: disable=unused-import
from . utils import debug_msg, err_msg, sys_msg
# pylint: enable=unused-import

__all__ = ["genlists", "measure"]


def genlists(low: int = LOWER, high: int = UPPER,
             size=(ITERATIONS, LENGTH_DEFAULT),
             dtype=int) -> np.ndarray:
    """
        NAME
            genlists
        DESCRIPTION
            Basically a wrapper around np.random.randint with
            default values provided.
    """
    # highly sorted data
    # data = []
    # for _ in range(size[0]):
    #    data.append(np.arange(MAXLENGTH)[::-1])
    # print(data)
    # return np.array(data)
    return np.random.randint(low, high, size, dtype)


# pylint: disable=too-many-locals
def measure(ldata: np.ndarray, algo: Callable,
            nlists: int = ITERATIONS, llength: int = 10,
            verbose: int = 0):
    """
        Measure algorithm performance in terms of execution time ("t"),
        number of comparisons ("comps"), and number of swaps ("swaps").

        Parameters
        ----------
            ldata : numpy.ndarray – A two-dimensional numpy array containing
                    lists of numbers (one list per row); the number of lists
                    (rows) must be >= nlists; the length of each list (number
                    of columns) must >= llength
            algo : Callable/function used to sort the lists
            nlists : int – Number of lists (number of rows in ldata) to process
            llength : int – List-length (number of columns in ldata)
            verbose : bool – if True, diagnostic/debug messages get printed

        Returns
        -------
        DataFrame({'algorithm': np.repeat(algo.__name__, nlists),
                   'length': np.repeat(llength, nlists),
                   't': t, 'comps': comps, 'swaps': swaps})

    """
    # Initialise three lists to contain measurement data
    t, comps, swaps = [], [], []
    try:
        if verbose:
            print(f"algorithm {algo.__name__}",
                  f"ldata.shape {ldata.shape}",
                  f"llength {llength}",
                  f"nlists {nlists}", sep="\n", end="\n\n")
        # Sanity check
        try:
            if llength > ldata.shape[1]:
                errormsg = "llength > (list length in "
                errormsg += "ldata ({ldata.shape[1]}))"
                raise IndexError(errormsg)
        except AttributeError as exception:
            err_msg(f"ldata should be a numpy.ndarray: {repr(exception)}")
            raise
        except IndexError as exception:
            err_msg(f"{repr(exception)}")
        # Measurements go here (number of measurements = nlists)
        for iteration in range(nlists):
            # Extract a list of suitable length from the data array
            lslice = ldata[iteration][:llength]
            if verbose > 1:
                print(f"List to be sorted: {lslice}")
            # quicksort() requires special handling
            # qs_algos = ('qsort', 'qsort2',
            #            'qsorti', 'qsorti2')
            # if algo.__name__ in qs_algos:
            # Measure number of comparisons and swaps
            _, c, s = algo(lslice, 0, len(lslice) - 1, copylist=True)
            # Measure execution time
            f = partial(algo, lslice, 0, len(lslice) - 1, copylist=True)
            timer = timeit.Timer(f)
            # Append results to data lists
            comps.append(c)
            swaps.append(s)
            t.append(timer.timeit(1))
        if verbose > 1:
            print(f"\nt = {t}, comps = {comps}",
                  f"swaps = {swaps}", sep="\n", end="\n\n")
        if verbose:
            print(f"t.mean = {np.mean(t) / 1e-3:01.4f} ± "
                  f"{np.std(t) / 1e-3:01.4f} ms")
            print(f"comps.mean = {np.mean(comps):01.2f} ± "
                  f"{np.std(comps):01.2f}")
            print(f"swaps.mean = {np.mean(swaps):01.2f} ± "
                  f"{np.std(swaps):01.2f}")
            print()
    except AttributeError as exception:
        err_msg("AttributeError:")
        err_msg(f"{repr(exception)}")
        raise
    except IndexError as exception:
        err_msg("IndexError:")
        err_msg(f"{repr(exception)}")
        raise
    # Create and return a dataframe with the measurements as columns
    return DataFrame({'algorithm': np.repeat(algo.__name__, nlists),
                      'length': np.repeat(llength, nlists),
                      't': t, 'comps': comps, 'swaps': swaps})
