import pandas as pd

# Load the dataset
df = pd.read_csv('Keywords_scopus_DOI.csv')

# Replace '|' with ';' in the 'Author Keywords' column
df['Author Keywords'] = df['Author Keywords'].str.replace(' | ', ';')

# Drop rows with missing values
df = df.dropna()

# Save the updated dataframe to a new CSV file
df.to_csv('updated_dataset.csv', index=False)
