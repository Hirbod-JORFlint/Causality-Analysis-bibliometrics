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
    A_num (int): The number of keywords to consider in each combination.

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
    co_occurrences.update(pair for keywords in df[keyword_column] if isinstance(keywords, list) for pair in combinations(keywords, A_num))

    # Convert the Counter object to a list of tuples and sort it
    co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)

    return co_occurrences[:100]

#top_100 = investigate_cooccurrence('diva_all.csv', 'Keywords', ';', 2)
#print(top_100)

df = pd.read_csv('diva_all.csv')

# Split the 'Keywords' column into a list of keywords
df['Keywords'] = df['Keywords'].str.split(';').apply(lambda x: [word.strip().lower() for word in x] if isinstance(x, list) else x)

# Create a Counter object to count the co-occurrences
co_occurrences = Counter()

# Iterate over the 'Keywords' column
for keywords in df['Keywords']:
    # Check if keywords is a list before proceeding
    if isinstance(keywords, list):
        # Use combinations to find all pairs of keywords
        for pair in combinations(keywords, 1):
            # Increment the count for this pair of keywords
            co_occurrences[pair] += 1

# Convert the Counter object to a list of tuples and sort it
co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)

print(co_occurrences[:10])

# Co-occurrences with conditions
co_occurrences_with_cond = [item for item in co_occurrences if "human-centered computing" in item[0]]
print(co_occurrences_with_cond)
