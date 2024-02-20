# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load dataset
df = pd.read_excel('IEEE VIS papers 1990-2022.xlsx')

# Ensure that 'Year' is of datetime type
#df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# 'AuthorKeywords' might contain multiple keywords separated by a delimiter
# Let's assume the delimiter is a comma
df['AuthorKeywords'] = df['AuthorKeywords'].str.split(',')

# Expand the DataFrame so there's one keyword per row
df = df.explode('AuthorKeywords')

# Remove any leading/trailing whitespaces
df['AuthorKeywords'] = df['AuthorKeywords'].str.strip().str.lower()
df = df.dropna(subset=['AuthorKeywords'])

# Making list of keywords
keywords = df['AuthorKeywords'].values
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df = df[['Year', 'AuthorKeywords']]
df.set_index('Year', inplace=True)
keyword_trends = df.groupby([pd.Grouper(freq='Y'), 'AuthorKeywords']).size().unstack(fill_value=0)

for keyword1 in keyword_trends.columns:
    for keyword2 in keyword_trends.columns:
        if keyword1 != keyword2:  # Exclude self-referential tests
            # Conduct Granger causality test
            test_result = grangercausalitytests(keyword_trends[[keyword1, keyword2]], maxlag=1, verbose=False)
            # Print results (change the significance level if needed)
            if test_result[1][0]['ssr_ftest'][1] < 0.05:  # Check p-value for significance
                print(f"Granger causality test between '{keyword1}' and '{keyword2}' is significant.")
            else:
                print(f"Granger causality test between '{keyword1}' and '{keyword2}' is not significant.")










print(1+'rr')

# Create a pivot table with years as columns, keywords as rows, and citation counts as values
pivot_df = df.pivot_table(values='AminerCitationCount', index='Year', columns='AuthorKeywords', fill_value=0)

# Replace citation counts with a binary value indicating whether the keyword was used that year
binary_df = pivot_df.applymap(lambda x: 1 if x > 0 else 0)

# Now we can apply the Granger causality test to each pair of keywords
# We'll store the results in a DataFrame
granger_results = pd.DataFrame(index=binary_df.columns, columns=binary_df.columns)
columns_list = binary_df.columns.tolist()

for keyword1 in columns_list:
    for keyword2 in columns_list:
        if keyword1 != keyword2:
            test_result = grangercausalitytests(binary_df[[keyword1, keyword2]], maxlag=2)
            p_values = [round(test_result[i+1][0]['ssr_ftest'][1],4) for i in range(2)]
            if any(p_value < 0.05 for p_value in p_values):
                granger_results.loc[keyword1, keyword2] = 1
            else:
                granger_results.loc[keyword1, keyword2] = 0

print(granger_results)

#binary_df.to_csv('binary_df.csv', index=True)
