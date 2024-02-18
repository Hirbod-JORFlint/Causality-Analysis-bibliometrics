# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Load your dataset
df = pd.read_excel('IEEE VIS papers 1990-2022.xlsx')

# Ensure that 'Year' is of datetime type
#df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# 'AuthorKeywords' might contain multiple keywords separated by a delimiter
# Let's assume the delimiter is a comma
df['AuthorKeywords'] = df['AuthorKeywords'].str.split(',')

# Expand the DataFrame so there's one keyword per row
df = df.explode('AuthorKeywords')

# Remove any leading/trailing whitespaces
df['AuthorKeywords'] = df['AuthorKeywords'].str.strip()


# Create a pivot table with years as columns, keywords as rows, and citation counts as values
pivot_df = df.pivot_table(values='AminerCitationCount', index='AuthorKeywords', columns='Year', fill_value=0)

# Replace citation counts with a binary value indicating whether the keyword was used that year
binary_df = pivot_df.applymap(lambda x: 1 if x > 0 else 0)

# Now we can apply the Granger causality test to each pair of keywords
# We'll store the results in a DataFrame
granger_results = pd.DataFrame(index=binary_df.index, columns=binary_df.index)

for keyword1 in binary_df.index:
    for keyword2 in binary_df.index:
        if keyword1 != keyword2:
            test_result = grangercausalitytests(binary_df[[keyword1, keyword2]], maxlag=2, verbose=False)
            p_values = [round(test_result[i+1][0]['ssr_ftest'][1],4) for i in range(2)]
            if any(p_value < 0.05 for p_value in p_values):
                granger_results.loc[keyword1, keyword2] = 1
            else:
                granger_results.loc[keyword1, keyword2] = 0

print(granger_results)

#binary_df.to_csv('binary_df.csv', index=True)
