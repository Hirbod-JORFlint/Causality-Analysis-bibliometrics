import pandas as pd
import itertools
from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

import pandas as pd
import itertools
from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def investigate_cooccurrence(filename, keyword_column, separator, A_num):
    # Load the dataset
    df = pd.read_csv(filename)

    # Drop rows with NaN values in the keyword column
    df = df.dropna(subset=[keyword_column])

    # Initialize a Counter object to count the co-occurrences
    cooccurrences = Counter()
    word_counts = Counter()
    original_cooccurrences = Counter()

    # Create stemmers for English and Swedish
    stemmer_en = SnowballStemmer('english')
    stemmer_sv = SnowballStemmer('swedish')

    # Get the stopwords for English and Swedish
    stopwords_en = set(stopwords.words('english'))
    stopwords_sv = set(stopwords.words('swedish'))

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Get the keywords from the keyword column and split them by the separator
        keywords = row[keyword_column].split(separator)

        # Strip whitespace and lowercase the keywords
        keywords = [keyword.strip().lower() for keyword in keywords]

        # Get all combinations of A_num keywords from the original keywords
        original_combinations = itertools.combinations(keywords, A_num)

        # Increment the count for each original combination
        for combination in original_combinations:
            original_cooccurrences[combination] += 1

        # Separate the words within a keyword
        words = []
        for keyword in keywords:
            words.extend(keyword.split())

        # Remove stopwords
        words = [word for word in words if word not in stopwords_en and word not in stopwords_sv]

        # Stem the words based on their detected language
        stemmed_words = []
        for word in words:
            if len(word) > 2 and any(char.isalpha() for char in word):  # Check if the word is long enough and contains any alphabetic characters
                try:
                    language = detect(word)
                    if language == 'en':
                        stemmed_words.append(stemmer_en.stem(word))
                    elif language == 'sv':
                        stemmed_words.append(stemmer_sv.stem(word))
                    else:
                        stemmed_words.append(word)
                except LangDetectException:
                    pass  # If language detection fails, just pass

        # Count the stemmed words
        word_counts.update(stemmed_words)

        # Get all combinations of A_num stemmed words
        combinations = itertools.combinations(stemmed_words, A_num)

        # Increment the count for each combination
        for combination in combinations:
            cooccurrences[combination] += 1

    # Combine cooccurrences and original_cooccurrences
    combined_cooccurrences = cooccurrences.copy()
    for key, value in original_cooccurrences.items():
        if key not in combined_cooccurrences:
            combined_cooccurrences[key] = value

    # Sort the co-occurrences and word counts in descending order of count and convert them to lists
    combined_cooccurrences = sorted(combined_cooccurrences.items(), key=lambda x: x[1], reverse=True)
    word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    return combined_cooccurrences, word_counts




def extract_unique_keywords(data):
    keywords = set()
    for item in data:
        keywords.update(k.strip() for k in item[0])
    return list(keywords)


top_100,oc2 = investigate_cooccurrence('MIT2.csv', 'Keywords', ';', 3)
print(top_100[:50])
print(extract_unique_keywords(top_100[:50]))
ex = extract_unique_keywords(top_100)
print('vuxnas lärande' in ex)



