import csv
from collections import Counter

def count_cooccurrences(filename):
    # Initialize a Counter to store the co-occurrence counts
    cooccurrences = Counter()

    # Read the file
    with open(filename, 'r',encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        lines = list(reader)

    # Flatten the list of lines and count occurrences
    words = [word for line in lines for word in line]
    cooccurrences = Counter(word for word in words if any(word in other_word for other_line in lines for other_word in other_line))

    # Sort the words in descending order of their counts
    sorted_words = sorted(cooccurrences.items(), key=lambda x: x[1], reverse=True)

    return sorted_words

# Use the function
sorted_words = count_cooccurrences("keywords_column.csv")

print(sorted_words[:20])
