"""= = = = = =
Exercise 3
= = = = = = """

import utils
from defaults import DEFAULTS

__all__ = ['exercise_3_main']

def exercise_3_main():
    """main of exercise 3"""
    print(__doc__)
    file_name = DEFAULTS['files']['fileName']
    name = None
    try:
        try:
            while not name:
                name = input("Name, please: ")
        except EOFError:
            pass
        with open(file_name, 'a', encoding="utf8") as file:
            file.write(f"{name}\n")
            print(f"Successfully appended text to {file_name}")
        utils.print_file_contents(file_name)
    except OSError as exception:
        print(f"OSError: {exception}")
        raise

if __name__ == "__main__":
    exercise_3_main()
