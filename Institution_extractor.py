import pandas as pd
import re
from collections import Counter


# Assuming df is DataFrame
df = pd.read_csv('df_processed.csv')  # replace '*.csv' with your file path

# Function to extract institute names
def extract_institute(name):
    if isinstance(name, str):
        institutes = re.findall(r'\((.*?)\)', name)
        return ', '.join(institutes)
    else:
        return ''

# Function to extract numbers within brackets(i.e ORG code)
def extract_numbers(institute):
    if isinstance(institute, str):
        numbers = re.findall(r'\[(\d+)\]', institute)
        return numbers
    else:
        return []
    
# Apply the function to the 'Name' column and create a new 'Institute' column
df['Institute'] = df['Name'].apply(extract_institute)

# Apply the function to the 'Institute' column and create a new 'Numbers' column
df['Numbers'] = df['Institute'].apply(extract_numbers)

# Flatten the list of lists
numbers_list = [num for sublist in df['Numbers'].tolist() for num in sublist]

# Count the occurrences of each number
counter = Counter(numbers_list)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_file.csv', index=False)  # replace 'updated_file.csv' with desired file path


# Find the top 20 most repeated numbers
top_20 = counter.most_common(20)

print(top_20)
