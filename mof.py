import json
import pandas as pd

# Load JSON data
with open('data\\tobmof-13511-(id_163037).json') as file:
    mof_data = json.load(file)

# Extracting relevant features
features_list = []
for mof in mof_data["mofs"]:
    features_list.append({
        'Name': mof['name'],
        'Void Fraction': mof['void_fraction'],
        'Surface Area (m2/g)': mof['surface_area_m2g'],
        'PLD': mof['pld'],
        'LCD': mof['lcd'],
        'Functional Groups': mof['functional_groups'],
        'Metal': mof['metal'],
        'Organic Linker': mof['organic_linker'],
        'Hydrogen Capacity': mof['hydrogen_capacity']
    })

# Convert to DataFrame
features_df = pd.DataFrame(features_list)

# Save the DataFrame to a CSV file
features_df.to_csv('mof_features.csv', index=False)
