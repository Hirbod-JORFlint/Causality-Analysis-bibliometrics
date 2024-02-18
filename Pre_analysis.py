import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load the data
df = pd.read_excel('IEEE VIS papers 1990-2022.xlsx')

# Display the first few rows of the dataframe
print(df.head())

# Count the number of papers per year
papers_per_year = df['Year'].value_counts().sort_index()
print('Papers per year:')
print(papers_per_year)

# Count the number of papers per conference
papers_per_conference = df['Conference'].value_counts()
print('Papers per conference:')
print(papers_per_conference)

# Analyze the citations
average_citations = df['CitationCount_CrossRef'].mean()
print('Average number of citations:', average_citations)

# Display the distribution of citations
plt.hist(df['CitationCount_CrossRef'], bins=20, edgecolor='black')
plt.title('Distribution of Citations')
plt.xlabel('Number of Citations')
plt.ylabel('Number of Papers')
plt.show()

# Find the papers with the most citations
top_cited_papers = df.sort_values('CitationCount_CrossRef', ascending=False).head(10)
print('Top 10 cited papers:')
print(top_cited_papers[['Title', 'Year', 'CitationCount_CrossRef']])

# Analyze how citations evolve over time
citations_over_time = df.groupby('Year')['CitationCount_CrossRef'].sum()
plt.plot(citations_over_time)
plt.title('Citations Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Citations')
plt.show()


# Split the 'AuthorNames-Deduped' column into individual authors
df['Authors'] = df['AuthorNames-Deduped'].str.split(';')

# Flatten the list of authors and count the occurrences of each author
authors = [author for sublist in df['Authors'] for author in sublist]
author_counts = Counter(authors)

# Display the 10 most published authors
most_published_authors = author_counts.most_common(10)
print('Most published authors:')
for author, count in most_published_authors:
    print(f'{author}: {count} papers')

# Analyze collaborations between authors
# This can be complex depending on how you define a collaboration, 
# but a simple approach is to count the number of co-authors for each paper
df['NumAuthors'] = df['Authors'].apply(len)
average_num_authors = df['NumAuthors'].mean()
print(f'Average number of authors per paper: {average_num_authors:.2f}')


#Does the number of authors of a paper cause a change in the number of citations?
#Does the length or complexity of an abstract cause a change in the number of citations? length has standards --keyword extraction
#
#Does the topic of the paper, inferred from the abstract, cause a change in the number of citations?
#Does the length of a paper (measured by the difference between LastPage and FirstPage) influence the number of internal references (InternalReferences)?
#Does the conference (Conference) cause a change in the number of authors (AuthorNames-Deduped)?**
