import os
import pandas as pd

df = pd.read_csv('climate_new/static_filtered.csv', dtype={'STAID': str})
ids = []

for staid in df['STAID']:
    ids.append(staid.zfill(8))

# Iterate over the files in the directory
for staid in ids:
    file_path = f'climate_new/{staid}.csv'

    # Check if the path is a file
    if os.path.isfile(file_path):
        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Remove the first 4 characters
        assert content[:4] == 'Date'
        modified_content = content[4:]

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(modified_content)