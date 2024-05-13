import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Check if the file exists
if os.path.exists('time_series_recent.csv'):
    # If the file exists, load the dataset
    time_df = pd.read_csv('time_series_recent.csv')
else:
    # If the file does not exist, perform the operations to create the dataset

    # Load the dataset
    time_df = pd.read_csv('final_df_processed.csv')

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

# Create a dictionary from the documents
dictionary = corpora.Dictionary(docs)

# Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) tuples
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Create an LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10)

# Print the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# Visualize the topics
vis = gensimvis.prepare(lda_model, corpus, dictionary, mds='tsne', n_jobs=-1)


log_likelihoods = []
for i in range(1, 51):  # for 50 iterations
    lda_model.update(corpus)
    log_likelihood = lda_model.log_perplexity(corpus)
    log_likelihoods.append(log_likelihood)

# Plot log likelihood per iteration
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), log_likelihoods)
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood per Iteration with LDA')
plt.grid()
plt.show()

# Save visualization to HTML
pyLDAvis.save_html(vis, 'lda_visualization.html')
