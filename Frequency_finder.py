import pandas as pd
from itertools import combinations
from collections import Counter

def investigate_cooccurrence(filename, keyword_column, separator, A_num):
    """
    Investigate the co-occurrence of keywords in a dataset.

    Parameters:
    filename (str): The name of the CSV file containing the dataset.
    keyword_column (str): The name of the column in the dataset that contains the keywords.
    separator (str): The character used to separate keywords in the keyword column.
    A_num (int): The number of keywords to consider in each combination(n_grams).

    Returns:
    list: A list of tuples where each tuple contains a combination of keywords and its count. 
          The list is sorted in descending order of count.
    """
    # Load your dataset
    df = pd.read_csv(filename)

    # Split the 'Keywords' column into a list of keywords and strip whitespaces
    df[keyword_column] = df[keyword_column].str.split(separator).apply(lambda x: [word.strip().lower() for word in x] if isinstance(x, list) else x)

    # Create a Counter object to count the co-occurrences
    co_occurrences = Counter()

    # Use list comprehension to find all pairs of keywords and increment their count
    co_occurrences.update(tuple(sorted(comb)) for keywords in df[keyword_column] if isinstance(keywords, list) for comb in combinations(set(keywords), A_num))


    # Convert the Counter object to a list of tuples and sort it
    co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)

    return co_occurrences


top_100 = investigate_cooccurrence('diva_all.csv', 'Keywords', ';', 2)
print(top_100[:20])

df = pd.read_csv('diva_all.csv')

# Keywords counting in data
df['Keyword_count'] = df['Keywords'].str.split(';').apply(lambda x: len(x) if isinstance(x, list) else 0)

# Split the 'Keywords' column into a list of keywords
df['Keywords'] = df['Keywords'].str.split(';').apply(lambda x: [word.strip().lower() for word in x] if isinstance(x, list) else x)


# Create a Counter object to count the co-occurrences
co_occurrences = Counter()

# Iterate over the 'Keywords' column
for keywords in df['Keywords']:
    if isinstance(keywords, list):
        for comb in combinations(set(keywords), 3):
            co_occurrences.update([tuple(sorted(comb))])

# Convert the Counter object to a list of tuples and sort it
co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)

print(co_occurrences[:5])

# Co-occurrences with conditions
co_occurrences_with_cond = [item for item in co_occurrences if "human-centered computing" in item[0]]
#print(co_occurrences_with_cond)

# Making Keyword list
dd = df['Keywords'].tolist()

# Finding Indices in any order
# combination
combination = set(['recognition of prior learning', 'validation', 'validering'])

# Check if all elements of the combination exist in each row
combination_exists = [combination.issubset(set(sublist)) if isinstance(sublist, list) else False for sublist in dd]

# Get the indices where the combination exists
indices = [i for i, x in enumerate(combination_exists) if x]

# Print the indices
print(indices)
print(len(indices))


# Finding Indices in the specific order(in 2-grams order does not matter)
# combination
combination = ['recognition of prior learning', 'validation', 'validering']

# Check if all elements of the combination exist in each row in the same order[preventing the TypeError]
combination_exists = [all(item in sublist for item in combination) if isinstance(sublist, list) else False for sublist in dd]

# Get the indices where the combination exists
indices = [i for i, x in enumerate(combination_exists) if x]

# Print the indices
print(indices)


# If we were to go through each row multiple times
# Iterate over each row
for i, sublist in enumerate(dd):
    # Check if sublist is a list before proceeding
    if isinstance(sublist, list):
        # Iterate over each combination in the row
        for comb in combinations(sublist, 3):
            # If the combination matches, store the index
            if set(comb) == combination:
                indices.append(i)

# Print the indices
print(indices)
print(len(indices))
