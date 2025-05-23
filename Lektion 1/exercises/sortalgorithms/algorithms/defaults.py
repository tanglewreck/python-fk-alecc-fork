"""
    NAME
        defaults.py
    DESCRIPTION
        Defaults for sort algorithms and performance measurements.
    AUTHOR
        mier
    DATE
        2025-02-19
    VERSION
        2025-05-05

"""
import numpy as np
# pylint: disable=unused-import
from . bubblesort import bubblesort
from . bubblesort import bubblesort2
from . bubblesort import bubblesort3
from . insertionsort import insertionsort
from . insertionsort import insertionsort2
from . insertionsort import insertionsort3
from . insertionsort import insertionsortwikipedia_for
from . insertionsort import insertionsortwikipedia_while
from . quicksort import quicksort, quicksort2, quicksort_iterative
# pylint: enable=unused-import

__all__ = ["ALGORITHMS", "ALGOSALL"]
__all__ += ["BS", "BSALL", "IS"]
__all__ += ["COLUMNS"]
__all__ += ["FIG_DIM", "FIG_DPI"]
__all__ += ["ITERATIONS"]
__all__ += ["LENGTH_DEFAULT", "LIST_LENGTHS"]
__all__ += ["LIST_LENGTHS_2"]
__all__ += ["LOWER", "UPPER"]
__all__ += ["NPREGEN", "MAXLENGTH"]
__all__ += ["SAVEPATH"]

# pylint: disable=wildcard-import, unused-wildcard-import, unused-import

# Directory where generated lists are saved:
SAVEPATH = "lists/"
LENGTH_DEFAULT = 10
# Plot size
FIG_DIM = (30, 15)
FIG_DPI = 100
#
# ALGORITHMS
BS1 = (bubblesort, )
BS2 = (bubblesort2, )
BS3 = (bubblesort3, )
BS = (bubblesort, bubblesort2, bubblesort3)
BSALL = (bubblesort, bubblesort2 )
#
IS1 = (insertionsort, )
IS2 = (insertionsort2, )
IS3 = (insertionsort3, )
IS = IS1 + IS2 + IS3
ISWFOR = (insertionsortwikipedia_for, )
ISWWHILE = (insertionsortwikipedia_while, )
ISW = ISWFOR + ISWWHILE
ISALL = IS + ISW
#
QS1 = (quicksort, )
QS2 = (quicksort2, )
QS3 = (quicksort_iterative, )
QS = QS1 + QS2 + QS3
ALGOSALL = BS + IS
ALGORITHMS = IS3 + ISW
ALGORITHMS = IS3 + QS
ALGORITHMS = BS + IS + ISW
ALGORITHMS = IS3 + QS
ALGORITHMS = IS3 + QS3
#
COLUMNS = ('t', 'comps', 'swaps')

# Number of iterations (=number of arrays to sort) during data-collection.
ITERATIONS = 10
ITERATIONS = 5
ITERATIONS = 2

# Min/max sizes of a random number
LOWER, UPPER = 1, 10_000
LOWER, UPPER = 1, 100

# Number of lists to pregenerate.
# This number sets an upper limit to ITERATIONS.
NPREGEN = 10_000
# The length of the pregenerated lists
MAXLENGTH = 20_000

# List-lenghts examples:
# LIST_LENGTHS = np.arange(10, 100, 10)
# LIST_LENGTHS = np.append(LIST_LENGTHS, np.arange(100, 301, 50))
# LIST_LENGTHS = np.append(np.arange(2,10) + np.arange(10, 100, 10))
# A somewhat complex case...
LIST_LENGTHS_2 = np.append(np.arange(10, 100, 10),
                           np.append(
                               np.append(
                                   np.append(np.arange(100, 200, 50),
                                             np.arange(200, 600, 100)),
                                   np.arange(600, 1200, 200)),
                               np.array([1500, 2000])))
# --> array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
#            150, 200, 300, 400, 500, 600, 800, 1000,
#            1500, 2000])
LIST_LENGTHS = np.append(np.arange(10, 110, 10),
                         np.array([150, 200, 250, 300, 350, 400, 450, 500, 750,
                                   1000, 1200]))
LIST_LENGTHS = np.array([50, 100, 200, 500, 750])
LIST_LENGTHS = np.array([1000, 2000, 3000, 4000, 5000])
LIST_LENGTHS = np.array([50, 100, 200, 500, 750])
LIST_LENGTHS = np.array([1_000, 2_000, 3_000, 5_000])
LIST_LENGTHS = np.array([100, 500, 1_000, 5_000, 10_000])
LIST_LENGTHS = np.array([10, 20, 30 , 40, 50, 60, 70, 80,
                         90, 100, 200, 300, 400, 500, 750,
                         1_000, 2_000, 5_000, 10_000])
