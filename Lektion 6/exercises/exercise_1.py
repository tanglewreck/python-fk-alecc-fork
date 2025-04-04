"""= = = = = =
Exercise 1
= = = = = = """

import utils
from defaults import DEFAULTS

__all__ = ['exercise_1_main']

def exercise_1_main():
    """main() of exercise 1"""
    print(__doc__)
    file_name = DEFAULTS['files']['fileName']
    text = DEFAULTS['texts']['text1']
    try:
        with open(file_name, 'w', encoding="utf8") as file:
            file.write(f"{text}\n")
            print(f"Successfully wrote to {file_name}")
        utils.print_file_contents(file_name)
    except OSError as exception:
        print(f"OSError: {exception}")
        raise


if __name__ == "__main__":
    exercise_1_main()
