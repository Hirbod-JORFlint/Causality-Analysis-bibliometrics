import pandas as pd
import dask.dataframe as dd
from collections import Counter
from itertools import combinations
from scipy.sparse import lil_matrix
import itertools
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
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

    # Create a Bayesian Network from the co-occurrences
    edges = [(a, b) for (a, b), _ in co_occurrences]
    model = BayesianModel(edges)

    # Prepare the data for the model
    unique_keywords = list(set(itertools.chain.from_iterable(keywords for keywords in df[keyword_column] if isinstance(keywords, tuple))))
    data = pd.DataFrame(0, index=np.arange(len(df)), columns=unique_keywords)
    for index, row in df.iterrows():
        if isinstance(row[keyword_column], tuple):
            for keyword in row[keyword_column]:
                if keyword in data.columns:
                    data.loc[index, keyword] = 1
    data.to_csv('gg.csv')
    # Fit the data to the model using Maximum Likelihood Estimator
    try:
        model.fit(data, estimator=MaximumLikelihoodEstimator)
    except KeyError as e:
        print(f"KeyError: {e}. The keyword may not exist in the dataset.")
    
    return model, co_occurrences




# Call the function with the appropriate parameters
model, co_occurrences = investigate_cooccurrence("MIT.csv", "Keywords", ";", 2)

# Print the co-occurrences
print(co_occurrences)


# Print Bayesian Network CPDs
#for node in model.nodes():
    #print(model.get_cpds(node))

# Print local dependencies for every nodes
# for node in model.nodes():
    #print(model.local_independencies(node))

# Perform inference on the model
#infer = VariableElimination(model)

# print(infer.query(variables=['chaos'], evidence={'design representations': 1}))
# Create a NetworkX graph from the Bayesian Model
G = nx.DiGraph()
G.add_edges_from(model.edges())

# Node size based on the degree of the node
node_sizes = [G.degree(node) * 100 for node in G.nodes]

# Node color based on the community
communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
community_map = {node: i for i, community in enumerate(communities) for node in community}
node_colors = [community_map.get(node) for node in G.nodes]

# Node labels
#labels = {node: node if node in ['artificial neural networks', 'evaluation'] else '' for node in G.nodes}

# Node labels
labels = {node: node for node in G.nodes}

# Node hover text
node_text = [labels[node] for node in G.nodes]

# Draw the graph
pos = nx.spring_layout(G, k=0.3)  # k regulates the distance between nodes
#nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", node_shape="o", alpha=0.5, linewidths=40)
#nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_sizes, node_color=node_colors, alpha=0.6, linewidths=0.1)

#plt.show()

# Create edges
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Create nodes
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=node_text,  # this line sets the hover text
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# Color node points by the number of connections.
node_adjacencies = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

fig.show()
