import pandas as pd
import itertools
from collections import Counter
from itertools import combinations
import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def investigate_cooccurrence(filename, keyword_column, separator, A_num):
    df = pd.read_csv(filename)

    # Convert lists of keywords to tuples
    # ignore empty strings
    #df[keyword_column] = df[keyword_column].str.split(separator).apply(lambda x: tuple(word.strip().lower() for word in x) if isinstance(x, list) else x)
    df[keyword_column] = df[keyword_column].str.split(separator).apply(lambda x: tuple(word.strip().lower() for word in x if word.strip()) if isinstance(x, list) else x)
    co_occurrences = Counter()
    co_occurrences.update(tuple(sorted(comb)) for keywords in df[keyword_column] if isinstance(keywords, tuple) for comb in combinations(set(keywords), A_num))
    co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)

    # Prepare the data for the model
    unique_keywords = list(set(itertools.chain.from_iterable(keywords for keywords in df[keyword_column] if isinstance(keywords, tuple))))
    data = pd.DataFrame(0, index=np.arange(len(df)), columns=unique_keywords)
    for index, row in df.iterrows():
        if isinstance(row[keyword_column], tuple):
            for keyword in row[keyword_column]:
                if keyword in data.columns:
                    data.loc[index, keyword] = 1
    data.to_csv('data.csv', index=False)

    # Create a BNLearner object
    learner = gum.BNLearner('data.csv')

    # Use a structure learning algorithm (default is MIIC)
    learner.useScoreBDeu()

    #print(learner)
    # Learn the Bayesian Network
    bn = learner.learnBN()
    
    return bn, co_occurrences

# Call the function with the appropriate parameters
model, co_occurrences = investigate_cooccurrence("MIT.csv", "Keywords", ";", 2)

# Print the co-occurrences
#print(co_occurrences)





