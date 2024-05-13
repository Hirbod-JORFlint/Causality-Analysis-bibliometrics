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
    df[keyword_column] = df[keyword_column].str.split(separator).apply(lambda x: tuple(word.strip().lower() for word in x) if isinstance(x, list) else x)
    co_occurrences = Counter()
    co_occurrences.update(tuple(sorted(comb)) for keywords in df[keyword_column] if isinstance(keywords, tuple) for comb in combinations(set(keywords), A_num))
    co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)
    co_occurrences = co_occurrences[:192]
    #print(co_occurrences)
    # Create an empty Bayesian Network
    bn = gum.BayesNet('Bayes')

    # Add nodes to the network
    # depending on the definition of unique_keywords the allocation might become impossible,
    # the code is modified to only consider the top
    unique_keywords = list(set(itertools.chain.from_iterable(keywords for keywords, _ in co_occurrences)))
    # Prepare the data for the model
    data = pd.DataFrame(0, index=np.arange(len(df)), columns=unique_keywords)
    for index, row in df.iterrows():
        if isinstance(row[keyword_column], tuple):
            for keyword in row[keyword_column]:
                if keyword in data.columns:
                    data.loc[index, keyword] = 1
    data.to_csv('data.csv', index=False)

    # Initialize a learner with data
    learner = gum.BNLearner('data.csv')

    # Use a structure learning algorithm (default is MIIC)
    learner.useScoreBDeu()

    # Learn the Bayesian Network
    bn = learner.learnBN()

    # Print the learned Bayesian Network
    #gnb.showBN(bn)

    return bn, co_occurrences, unique_keywords

# Call the function with the appropriate parameters
model2, co_occurrences2, words2 = investigate_cooccurrence("MIT.csv", "Keywords", ";", 2)

# Print the co-occurrences
#print(co_occurrences)





