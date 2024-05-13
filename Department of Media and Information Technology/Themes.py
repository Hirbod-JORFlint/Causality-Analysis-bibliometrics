import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import logging
import re
from itertools import combinations
from gensim.matutils import hellinger
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split

# Set up logging
#logging.basicConfig(filename='gensim.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Create the directory if it doesn't exist
if not os.path.exists('LDA_res'):
    os.makedirs('LDA_res')

def compute_stats(hellinger_distances):
    stats = []
    for sublist in hellinger_distances:
        min_val = np.min(sublist)
        max_val = np.max(sublist)
        mean_val = np.mean(sublist)
        median_val = np.median(sublist)
        std_dev = np.std(sublist)
        quantiles = np.percentile(sublist, [25, 50, 75])
        stats.append({
            'min': min_val,
            'max': max_val,
            'mean': mean_val, 
            'median': median_val, 
            'std_dev': std_dev, 
            '1st_quantile': quantiles[0], 
            '2nd_quantile': quantiles[1], 
            '3rd_quantile': quantiles[2]
        })
    return stats

# Check if the file exists
# if os.path.exists('time_series_recent.csv'):
#     # If the file exists, load the dataset
#     time_df = pd.read_csv('time_series_recent.csv')

# If the file does not exist, perform the operations to create the dataset

# Load the dataset
time_df = pd.read_csv('MIT.csv')

time_df = time_df.dropna(subset=['Keywords'])

# Lowercase the keywords and split them into a list
time_df['Keywords'] = time_df['Keywords'].str.split(';').apply(lambda x: [word.strip().lower() for word in x])

# Explode the 'Keywords' column into multiple rows
time_df_exploded = time_df.explode('Keywords')

# Count the occurrences of each keyword for each year
df_keyword_counts = time_df_exploded.groupby(['Year', 'Keywords']).size().unstack(fill_value=0)

# Filter the DataFrame to include only the keywords that have a count of 0 for the years before 2019
df_before_2019 = df_keyword_counts[df_keyword_counts.index < 2019]
keywords_before_2019 = df_before_2019.columns[(df_before_2019 == 0).all()]

# Filter the DataFrame to include only the keywords that have a count greater than 0 for the years from 2019 onwards
df_from_2019 = df_keyword_counts[df_keyword_counts.index >= 2019]
keywords_from_2019 = df_from_2019.columns[(df_from_2019 > 0).any()]

# Get the intersection of the two sets of keywords
filtered_keywords = keywords_before_2019.intersection(keywords_from_2019)

# Filter the original DataFrame to include only these keywords
time_df['Keywords'] = time_df['Keywords'].apply(lambda x: list(set(x).intersection(filtered_keywords)))

# Remove the rows where 'Keywords' column is empty
time_df = time_df[time_df['Keywords'].apply(lambda x: len(x) > 0)]

# Saving the time series corresponding to the recent years(2019 onwards)
time_df.to_csv('time_series_recent.csv')

# Convert your pandas series to a list of documents
docs = time_df['Keywords'].tolist()

# Split your documents into a training set and a test set
#train_docs, test_docs = train_test_split(docs, test_size=0.2, random_state=23)

# Create a dictionary from the documents
#dictionary = corpora.Dictionary(train_docs)

# There is no label
dictionary = corpora.Dictionary(docs)

# Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) tuples
#train_corpus = [dictionary.doc2bow(doc) for doc in train_docs]

# Convert test documents into the bag-of-words format
#test_corpus = [dictionary.doc2bow(doc) for doc in test_docs]

# Convert documents into the bag-of-words format
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Initialize an empty DataFrame to store the results
results = pd.DataFrame()
hellinger_distances = []

# Iterate over a range of different numbers of topics
for num_topics in range(5, 100):
    #print(num_topics)
    # Create an LDA model
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=40, iterations=5000, random_state=2, eval_every=50)

    # Compute Perplexity
    perplexity = lda_model.log_perplexity(corpus)

    # Compute Coherence Score
    #coherence_model_lda = CoherenceModel(model=lda_model, texts=test_docs, dictionary=dictionary, coherence='c_v')
    #coherence = coherence_model_lda.get_coherence()

    # Compute Topic Difference
    topic_diff = lda_model.diff(lda_model, distance='jaccard', num_words=50)
    topic_diff_matrix = np.array(topic_diff[0], dtype=float)
    
    # Compute Hellinger Distance
    #hellinger_dist = hellinger(lda_model.get_topics()[0], lda_model.get_topics()[1])
    
    topic_distributions = lda_model.get_topics()
    hellinger_dist = [hellinger(topic_distributions[i], topic_distributions[j]) for i, j in combinations(range(len(topic_distributions)), 2)]
    hellinger_distances.append(hellinger_dist)
    
    # Append the results to the DataFrame
    # Create a DataFrame from the dictionary and append it to the results DataFrame
    df_new = pd.DataFrame([{
        'Num_Topics': num_topics,
        'Perplexity': perplexity,
        'Topic_Difference': topic_diff,
    }], columns=['Num_Topics', 'Perplexity', 'Topic_Difference'])

    results = pd.concat([results, df_new], ignore_index=True)

    # Plotly heatmap
    # Get the top n keywords for each topic(n=2)
    keywords = [[word for word, _ in lda_model.show_topic(topicid, topn=5)] for topicid in range(lda_model.num_topics)]
    fig = go.Figure(data=go.Heatmap(
    z=topic_diff_matrix,
    x=keywords,
    y=keywords,
    colorscale='Viridis'
    ))
    fig.write_html(f"LDA_res/plotly_heatmap_{num_topics}.html")
    
    # Matplotlib heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(topic_diff_matrix, cmap='viridis')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Topic Difference')
    plt.title(f'Heatmap for {num_topics} Topics')
    plt.savefig(f"LDA_res/matplotlib_heatmap_{num_topics}.png")  # Save the figure
    plt.close()  # Close the figure

# Plot the results
plt.figure(figsize=(15, 10))


plt.plot(results['Num_Topics'].to_numpy(), results['Perplexity'].to_numpy())
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.title('Perplexity vs Number of Topics')
plt.show()
plt.close()

plt.boxplot(hellinger_distances)
plt.xlabel('Hellinger Distance Per Iterations')
plt.title('Boxplot of Hellinger Distances')

plt.tight_layout()
plt.show()

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(stats)

# Save the DataFrame to a CSV file
df.to_csv('stats.csv', index=False)

stats = compute_stats(hellinger_distances)
for i, stat in enumerate(stats):
    print(f"Iteration {i+1}: Min = {stat['min']}, Max = {stat['max']}, Mean = {stat['mean']}, Median = {stat['median']}, Std Dev = {stat['std_dev']}, 1st Quantile = {stat['1st_quantile']}, 2nd Quantile = {stat['2nd_quantile']}, 3rd Quantile = {stat['3rd_quantile']}")


# Save visualization to HTML
vis = gensimvis.prepare(lda_model, corpus, dictionary, mds='tsne', n_jobs=-1)
pyLDAvis.save_html(vis, 'lda_visualization.html')
