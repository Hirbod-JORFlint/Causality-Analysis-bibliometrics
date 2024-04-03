import pandas as pd
import itertools
from collections import Counter
from nltk.util import ngrams


def investigate_cooccurrence(filename, keyword_column, separator, A_num):
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Initialize a counter
    cooccurrence_counter = collections.Counter()
    
    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        # Skip rows where the keyword column is NaN
        if pd.isnull(row[keyword_column]):
            continue
        
        # Get the keywords for this row
        keywords = str(row[keyword_column]).split(separator)
        
        # Remove leading and trailing whitespaces and convert to lower case
        keywords = [keyword.strip().lower() for keyword in keywords]
        
        # Generate the n-grams for these keywords
        keyword_ngrams = list(ngrams(keywords, A_num))
        
        # Update the counter with these n-grams
        cooccurrence_counter.update(keyword_ngrams)
    
    # Convert the counter to a list of tuples and sort it in descending order of count
    cooccurrence_list = sorted(cooccurrence_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Create a list of unique simple keywords
    simple_keywords = set([keyword for keyword_list in df[keyword_column].apply(lambda x: str(x).split(separator) if pd.notnull(x) else []) for keyword in keyword_list])
    
    # Filter the cooccurrence_list to only include combinations where all keywords are in the simple_keywords list
    cooccurrence_list = [(combination, count) for combination, count in cooccurrence_list if all('sverige' in keyword for keyword in combination)]
    
    return cooccurrence_list


def investigate_cooccurrence(filename, keyword_column, separator, A_num):
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Drop rows with missing values in the keyword column
    df = df.dropna(subset=[keyword_column])
    
    # Strip whitespace, lowercase the keywords, and split them
    df[keyword_column] = df[keyword_column].str.strip().str.lower().str.split(separator)
    
    # Flatten the list of keywords
    keywords = list(itertools.chain.from_iterable(df[keyword_column].tolist()))
    
    # Generate all combinations of keywords
    combinations = list(itertools.combinations(keywords, A_num))
    
    # Initialize a Counter for the combinations
    count = Counter()
    
    # Iterate over each row in the keyword column
    for row in df[keyword_column]:
        # Generate n-grams from the row
        row_ngrams = list(ngrams(row, A_num))
        
        # Increment the count for each combination that is a subset of the row's n-grams
        for combination in combinations:
            if any(set(combination).issubset(set(ngram)) for ngram in row_ngrams):
                count[combination] += 1
    
    # Sort the combinations by count in descending order and convert to list of tuples
    sorted_combinations = sorted(count.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_combinations


def extract_unique_keywords(data):
    keywords = set()
    for item in data:
        keywords.update(item[0])
    return list(keywords)


top_100 = investigate_cooccurrence('updated_file.csv', 'Keywords', ';', 1)
print(top_100[:50])
print(extract_unique_keywords(top_100[:50]))
ex = extract_unique_keywords(top_100)
print('vuxnas lärande' in ex)



