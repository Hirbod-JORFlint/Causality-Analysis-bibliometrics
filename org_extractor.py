import pandas as pd

# Load the dataset
df = pd.read_csv('updated_file.csv')

# Define a function to check if a row contains the numbers
def contains_target_numbers(row):
    # Convert the string representation of the list to a list
    numbers = eval(row['Numbers'])
    # Check if '###' or '###' is in the list
    #Department of math
    #return '307' in numbers or '11103' in numbers or '11104' in numbers or '2328' in numbers or '2327' in numbers or '884511' in numbers or '2326' in numbers or '2329' in numbers
    #Media and information tech
    return '6850' in numbers
# Apply the function to each row
df['ContainsTargetNumbers'] = df.apply(contains_target_numbers, axis=1)

# Keep only the rows that contain the target numbers
df = df[df['ContainsTargetNumbers']]

# Save the filtered dataframe to a new CSV file
df.to_csv('MIT.csv', index=False)
