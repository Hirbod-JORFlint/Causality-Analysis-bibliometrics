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
    
    # Load the dataset
    df = pd.read_csv(filename)
    
    # Split the keywords and create combinations
    df[keyword_column] = df[keyword_column].apply(lambda x: list(combinations([word.strip() for word in x.split(separator)], A_num)) if isinstance(x, str) else [])
    
    # Flatten the list of keyword combinations and count the occurrences
    keyword_combinations = [item for sublist in df[keyword_column].tolist() for item in sublist]
    counter = Counter(keyword_combinations)
    
    # Get the top 15 most common combinations
    top_100 = counter.most_common(100)
    
    return top_100

top_100 = investigate_cooccurrence('diva_all.csv', 'Keywords', ';', 3)
for combo, count in top_100:
    print(f"Combination: {combo}, Count: {count}")
