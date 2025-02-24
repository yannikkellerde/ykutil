import os
import re
import random
import string


def bulk_rename(directory, pattern, replacement):
    """
    Renames files in the specified directory by replacing occurrences of the given
    pattern in the filename with the replacement string.

    Parameters:
        directory (str): The path to the folder containing the files to rename.
        pattern (str): The regex pattern to match in filenames.
        replacement (str): The string to replace the matched pattern.
    """

    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Compile the regex pattern
    regex = re.compile(pattern)

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Only match files that fit the specified pattern
        if regex.search(filename):
            # Construct the new filename
            new_filename = regex.sub(replacement, filename)

            # Get the full paths for renaming
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # Rename the file
            try:
                os.rename(old_path, new_path)
                print(f"Renamed '{filename}' to '{new_filename}'")
            except Exception as e:
                print(f"Error renaming '{filename}': {e}")


def random_string(length, capital=True, numbers=False, special=False):
    """
    Generates a random string of the specified length.

    Parameters:
        length (int): The length of the random string to generate.
        capital (bool): If True, include uppercase letters. Default is True.
        numbers (bool): If True, include digits. Default is False.
        special (bool): If True, include special characters. Default is False.

    Returns:
        str: The generated random string.
    """
    characters = string.ascii_lowercase
    if capital:
        characters += string.ascii_uppercase
    if numbers:
        characters += string.digits
    if special:
        characters += string.punctuation

    return "".join(random.choice(characters) for _ in range(length))
