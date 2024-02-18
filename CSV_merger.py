import pandas as pd


# merge the datasets
files = ['0.csv', '1.csv', '10.csv', '11.csv', '12.csv', '13.csv', '14.csv', '15.csv', '2.csv', '3.csv', '4.csv', '5.csv', '6.csv', '7.csv', '8.csv', '9.csv']

# Read each file into a pandas DataFrame and store all DataFrames in a list
dfs = [pd.read_csv(file) for file in files]

# Concatenate all DataFrames in the list into one DataFrame
merged_df = pd.concat(dfs)

# Write the merged DataFrame to a new file
merged_df.to_csv('merged_dataset.csv', index=False)
