import pandas as pd

# Load the datasets
df_updated = pd.read_csv('updated_dataset.csv')
df_missed = pd.read_csv('df_with_Name.csv')

# Initialize a list to store the updated keywords
updated_keywords = []

# Iterate through the rows of df_missed
for index, row in df_missed.iterrows():
    # Find corresponding row in df_updated based on 'DOI' and 'Title' columns
    matching_rows = df_updated[(df_updated['DOI'] == row['DOI']) & (df_updated['Title'] == row['Title'])]
    
    # Check if matching_row is not empty and 'Keywords' is missing in df_missed
    if not matching_rows.empty and pd.isnull(row['Keywords']):
        # Append the 'Author Keywords' from df_updated to the list
        updated_keywords.append(matching_rows['Author Keywords'].iloc[0])
    else:
        # If no update is needed, keep the original keyword
        updated_keywords.append(row['Keywords'])

# Update the 'Keywords' column in df_missed with the updated_keywords list
df_missed['Keywords'] = updated_keywords

# Save the updated dataframe to a new CSV file
df_missed.to_csv('final_dataset.csv', index=False)
