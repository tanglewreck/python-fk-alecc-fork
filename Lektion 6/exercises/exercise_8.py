"""= = = = = =
Exercise 8
= = = = = = """
# Radera din fil "test.txt"

import os
from defaults import DEFAULTS

__all__ = ['exercise_8_main']

def exercise_8_main():
    """main of exercise 8"""
    print(__doc__)
    file_name = DEFAULTS['files']['fileName']
    try:
        os.unlink(file_name)
        print(f"Successfully unlinked {file_name}")
    except OSError as exception:
        print(f"{exception}")

if __name__ == "__main__":
    exercise_8_main()
