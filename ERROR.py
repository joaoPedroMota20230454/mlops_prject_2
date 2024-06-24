import os
import fnmatch

def find_substring_in_files(directory, substring):
    for root, dirs, files in os.walk(directory):
        
        if root.startswith('.\\env'):
            continue
        
        for extension in ['*.py', '*.yml']:
            for filename in fnmatch.filter(files, extension):
                filepath = os.path.join(root, filename)

                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        contents = file.read()
                        if substring in contents:
                            print(f"Substring '{substring}' found in file: {filepath}")
                except Exception as e:
                    print(f"Could not read file {filepath}: {e}")

# Replace 'your_directory' with the path to your directory
directory_to_search = '.'
substring_to_find = 'fmppo'

find_substring_in_files(directory_to_search, substring_to_find)
