import pandas as pd
from collections import Counter
from itertools import combinations
import itertools
import numpy as np
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.image as gumimage
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def investigate_cooccurrence(filename, keyword_column, separator, A_num):
    df = pd.read_csv(filename)

    # Convert lists of keywords to tuples
    df[keyword_column] = df[keyword_column].str.split(separator).apply(lambda x: tuple(word.strip().lower() for word in x) if isinstance(x, list) else x)
    co_occurrences = Counter()
    co_occurrences.update(tuple(sorted(comb)) for keywords in df[keyword_column] if isinstance(keywords, tuple) for comb in combinations(set(keywords), A_num))
    co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)
    co_occurrences = co_occurrences[:30]
    #co_occurrences = [(tuple(k.replace('_', '') for k in keywords), count) for keywords, count in co_occurrences]
    #co_occurrences = [(tuple(k.replace(' ', '') for k in keywords), count) for keywords, count in co_occurrences]
    #co_occurrences = [(tuple(k.replace('-', '') for k in keywords), count) for keywords, count in co_occurrences]
    #print(co_occurrences)
    # Create an empty Bayesian Network
    bn = gum.BayesNet('Bayes')

    # Add nodes to the network
    # depending on the definition of unique_keywords the allocation might become impossible,
    # the code is modified to only consider the top
    unique_keywords = list(set(itertools.chain.from_iterable(keywords for keywords, _ in co_occurrences)))
    #print(unique_keywords)
    bn_list = []
    for keyword in unique_keywords:
        var = gum.LabelizedVariable(keyword, keyword, 2)
        #var.changeLabel(0, '0')
        #var.changeLabel(1, '1')
        bn_list.append(bn.add(var))


    # Add edges based on the co-occurrences
    edges = [(a, b) for (a, b), _ in co_occurrences]
    for edge in edges:
        bn.addArc(bn.idFromName(edge[0]), bn.idFromName(edge[1]))
    # Prepare the data for the model
    data = pd.DataFrame(0, index=np.arange(len(df)), columns=unique_keywords)
    for index, row in df.iterrows():
        if isinstance(row[keyword_column], tuple):
            for keyword in row[keyword_column]:
                if keyword in data.columns:
                    data.loc[index, keyword] = 1
    data.to_csv('data.csv', index=False)

    node_names = [bn.variableFromName(i).name() for i in bn.names()]
    for id in range(bn.size()):
        print(f"ID: {id}, Node Name: {bn.variable(id).name()}")
    print(node_names)
    print(bn)
    # Learn the Bayesian Network
    bn.generateCPTs()
    #gum.saveBN(bn,"Garage.bif")
    learner = gum.BNLearner('data.csv', bn)
    learner = learner.useSmoothingPrior(weight=100)
    bn = learner.learnParameters(bn.dag())

    # Print the learned Bayesian Network
    #gnb.showBN(bn)

    return bn, co_occurrences, unique_keywords

model, co_occurrences, words = investigate_cooccurrence("MIT.csv", "Keywords", ";", 2)
