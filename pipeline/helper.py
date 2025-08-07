import os
import importlib.util
from datetime import datetime
import random


def load_model_module(model_name):
    model_path = os.path.join("vision_models", model_name, f"{model_name}.py")
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_prompt(dataset_name):
    with open(f"dataset/{dataset_name}/{dataset_name}_prompt.txt") as f:
        prompt = f.readlines()
    # NOTE: readlines returns a list, hence take the first index assuming it is having prompt in one line
    return prompt[0]


def get_timestamped_filename(prefix, model_name, dataset_name, extension):
    now = datetime.now()
    timestamp = now.strftime("%H_%M_%d_%m_%Y")
    return f"{prefix}_{model_name}_{timestamp}.{extension}"


def get_filename_class_mapping(filename):
    """
    Reads a file where each line contains a key and a value separated by a space,
    and returns a dictionary.
    Args:
        filename (str): The path to the input file.
    Returns:
        dict: A dictionary where the first part of each line is the key
              and the second part is the value.
    """
    result_dict = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                # Remove leading/trailing whitespace and split the line by the first space
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    key = parts[0]
                    value = parts[1]
                    result_dict[key] = value
                else:
                    print(f"Skipping malformed line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return result_dict


def get_png_files_in_folder(folder_path):
    """
    Returns a list of all PNG files found directly within the specified folder.

    Args:
        folder_path (str): The path to the folder to scan.

    Returns:
        list: A list of filenames (strings) ending with '.png'.
              Returns an empty list if the folder does not exist or contains no PNGs.
    """
    png_files = []
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found or is not a directory.")
        return []

    try:
        # List all entries in the directory
        for item in os.listdir(folder_path):
            # Construct the full path to check if it's a file
            item_path = os.path.join(folder_path, item)
            # Check if it's a file and ends with .png (case-insensitive)
            if os.path.isfile(item_path) and item.lower().endswith(".png"):
                png_files.append(item)
    except Exception as e:
        print(f"An error occurred while reading folder '{folder_path}': {e}")
        return []

    return sorted(png_files)


def shuffle_list(lst):
    """Shuffles the input list in-place and returns it."""
    random.shuffle(lst)
    return lst

import time

def shuffle_class_name_in_prompt(input_string):
    """
    Splits a string, processes the second part, and returns a new concatenated string.

    Args:
        input_string: A string in the format "first_element: second_element".

    Returns:
        A concatenated string of the first element and the processed second element.
    """
    # Split the input string by ':' and take the first two elements.
    # The split() method handles cases where there might be more than one ':'.
    parts = input_string.split(':', 1)
    
    # Ensure there are at least two parts to avoid an IndexError.
    if len(parts) < 2:
        return "Invalid input format. Expected 'first:second'."
    
    first_element = parts[0].strip()
    second_element = parts[1].strip()
    
    # Split the second element by space. The split() method without arguments
    # handles multiple spaces between words and leading/trailing spaces.
    words = second_element.split()
    
    words = shuffle_list(words)
    
    # Join the list of words back into a single string.
    processed_string = "".join(words)
    
    # Concatenate the first element and the processed second element.
    return first_element + processed_string

