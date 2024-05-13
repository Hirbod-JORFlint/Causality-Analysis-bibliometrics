from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import combinations
from collections import Counter

def investigate_cooccurrence(filename, keyword_column, separator, A_num):
    """
    Investigate the co-occurrence of keywords in a dataset.

    Parameters:
    filename (str): The name of the CSV file containing the dataset.
    keyword_column (str): The name of the column in the dataset that contains the keywords.
    separator (str): The character used to separate keywords in the keyword column.
    A_num (int): The number of keywords to consider in each combination(n_grams).

    Returns:
    list: A list of tuples where each tuple contains a combination of keywords and its count. 
          The list is sorted in descending order of count.
    """
    # Load your dataset
    df = pd.read_csv(filename)

    # Split the 'Keywords' column into a list of keywords and strip whitespaces
    df[keyword_column] = df[keyword_column].str.split(separator).apply(lambda x: [word.strip().lower() for word in x] if isinstance(x, list) else x)

    # Create a Counter object to count the co-occurrences
    co_occurrences = Counter()

    # Use list comprehension to find all pairs of keywords and increment their count
    co_occurrences.update(tuple(sorted(comb)) for keywords in df[keyword_column] if isinstance(keywords, list) for comb in combinations(set(keywords), A_num))


    # Convert the Counter object to a list of tuples and sort it
    co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)

    return co_occurrences

coe = investigate_cooccurrence('IDA.csv', 'Keywords', ';', 2)[:120]
coe_set = set([tuple(sorted(pair)) for pair, _ in coe])

# Function for performing Augmented Dickey-Fuller
def perform_adf_test(series):
    result = adfuller(series)
    # Return the conclusion based on the p-value
    return result[1] < 0.05

def perform_granger_test(df, pairs, maxlag=5):
    """
    Perform the Granger causality test on a list of pairs.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the time series data.
    pairs (list of tuple): The list of pairs to test.
    maxlag (int): The maximum lag to consider for the test.

    Returns:
    results (dict): A dictionary where the keys are the pairs and the values are the test results.
    """
    results = []
    for pair in pairs:
        print('\n')
        print(pair)
        # Drop NaN values
        df_clean = df[list(pair)].dropna()
        # Add a small amount of noise to the data
        df_clean = df_clean + np.random.normal(0, 0.001, size=df_clean.shape)
        
        try:
            test_result = grangercausalitytests(df_clean, maxlag=maxlag)
            p_values = [round(test_result[i+1][0]['ssr_ftest'][1],4) for i in range(maxlag)]
            results.append([pair[0], pair[1]] + p_values)
        except ValueError:
            results.append([pair[0], pair[1]] + ['Test Infeasible']*maxlag)

    # Convert the results to a DataFrame
    columns = ['Keywords1', 'Keywords2'] + ['lag'+str(i+1) for i in range(maxlag)]
    results_df = pd.DataFrame(results, columns=columns)

    # Save the DataFrame to a CSV file
    results_df.to_csv('granger_test_results.csv')

    return results

# Load the data
df_new = pd.read_csv('time_series.csv')

# Save a copy of 'Year' column
year_column = df_new['Year'].copy()

# Drop constant columns
df_new = df_new.loc[:, (df_new != df_new.iloc[0]).any()]

# Initialize lists to store column names
stationary_cols = []
non_stationary_cols = []

# Perform ADF test on each column
for col in df_new.columns:
    if col != 'Year':  # Exclude 'Year' column
        if perform_adf_test(df_new[col]):
            stationary_cols.append(col)
        else:
            non_stationary_cols.append(col)

# Perform differencing on non-stationary columns
for col in non_stationary_cols:
    df_new[col] = df_new[col].diff()
    # After differencing, check if column became stationary
    if perform_adf_test(df_new[col].dropna()):  # dropna() as diff() introduces NaN values
        stationary_cols.append(col)
        non_stationary_cols.remove(col)

print("Stationary columns:", stationary_cols)
print("Non-stationary columns:", non_stationary_cols)

# Initialize an empty list to store stationary pairs
valid_pairs = []

# Check each pair in coe_set
for pair in coe_set:
    # If both elements of the pair are in the stationary list, add the pair to valid_pairs
    if pair[0] in stationary_cols and pair[1] in stationary_cols:
        valid_pairs.append(pair)

print("Valid pairs that are stationary:", valid_pairs)
reversed_pairs = [(b, a) for a, b in valid_pairs]



res=perform_granger_test(df_new,valid_pairs[:30])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#res2=perform_granger_test(df_new,reversed_pairs)
print(len(res))

# Add the 'Year' column back to the final dataset
df_new['Year'] = year_column

# Print the list of significant pairs
print("\nSignificant pairs:", significant_pairs)

# Function to plot the time series of the data in significant_pairs
def plot_time_series(df, pairs):
    # Create a directory to save the plots
    if not os.path.exists('granger'):
        os.makedirs('granger')

    for pair in pairs:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Year'].values, df[pair[0]].values, marker='o', label=pair[0])
        plt.plot(df['Year'].values, df[pair[1]].values, marker='o', label=pair[1])
        plt.title(f"Time series of {pair[0]} and {pair[1]}")
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend()
        # Save the plot in the 'granger' directory
        plt.savefig(f"granger/{pair[0]}_{pair[1]}.png")
        plt.close()


# Call the function to plot the time series of the data in significant_pairs
plot_time_series(df_new, significant_pairs)
