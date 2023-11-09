import os
import json
import pandas as pd

# Directory containing JSON files
json_directory = r'C:\Users\91843\Documents\Hydrogen regression\tobacco hydrogen'

# List to store data from JSON files
data_list = []

# Specify the columns you want to keep
selected_columns = ['void_fraction', 'surface_area_m2g', 'pld', 'lcd', 'functional_groups', 'hydrogen_capacity']

# Iterate over files in the directory
for filename in os.listdir(json_directory):
    if filename.startswith('tobmof') and filename.endswith('.json'):
        # Construct the full path to the JSON file
        json_file_path = os.path.join(json_directory, filename)
        
        # Read the JSON file
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            # Select only the desired columns
            selected_data = {key: data[key] for key in selected_columns}
            data_list.append(selected_data)

# Convert the list of selected JSON data into a DataFrame
df = pd.DataFrame(data_list)

# Specify the output CSV file path
csv_file_path = 'output.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print(f'Converted {len(data_list)} JSON files to {csv_file_path}')
