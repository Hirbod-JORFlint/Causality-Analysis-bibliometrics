import pandas as pd

# Load the dataset
df_processed = pd.read_csv('df_processed.csv')

# Remove rows with empty cells in 'keywords' column
df_processed = df_processed[df_processed['Keywords'].notna()]

# Save the new DataFrame to a new file
df_processed.to_csv('nd_processed.csv', index=False)
