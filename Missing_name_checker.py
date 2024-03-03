import pandas as pd

# Load your dataset
df = pd.read_csv('diva_all.csv')  # replace with your filename



# Find the rows that contain DOI values
df_with_name = df[df['Name'].notna() & (df['Name'] != '')]


# Save the target to csv
df_with_name.to_csv('df_with_Name.csv', index=False)


