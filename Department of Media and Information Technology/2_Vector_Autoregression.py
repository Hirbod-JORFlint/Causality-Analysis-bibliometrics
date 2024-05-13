import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
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

coe = investigate_cooccurrence('MIT.csv', 'Keywords', ';', 2)

# Load the dataset
df = pd.read_csv("time_series.csv")

# Drop the 'Year' column
df.drop(columns=['Year'], inplace=True)

# Drop constant columns
df = df.loc[:, df.apply(lambda x: len(x.unique()) > 1)]

# Perform stationary testing using the critical value and store the stationary variables
stationary_columns = []
for column in df.columns:
    result = adfuller(df[column], regression='ct')
    test_statistic = result[0]
    critical_value_5pct = result[4]['5%']
    if test_statistic < critical_value_5pct:
        stationary_columns.append(column)

# Define the significance level
significance_level = 0.05

# Perform stationary testing and store the stationary variables
stationary_columns_p_value = []
for column in df.columns:
    result = adfuller(df[column], regression='c')
    p_value = result[1]
    if p_value < significance_level:
        stationary_columns_p_value.append(column)



# Sort the columns
stationary_columns = sorted(stationary_columns)
stationary_columns_p_value = sorted(stationary_columns_p_value)

# Find the columns that exist in one list but not in the other
unique_to_original = [column for column in stationary_columns if column not in stationary_columns_p_value]
unique_to_p_value = [column for column in stationary_columns_p_value if column not in stationary_columns]

# Print the results
print("Columns that passed the original ADF test but not the p-value test:", unique_to_original)
print("Columns that passed the p-value ADF test but not the original test:", unique_to_p_value)

# Generate all possible pairs of stationary variables
pairs = list(combinations(stationary_columns_p_value, 2))

# Perform Vector autoregression on the stationary columns two by two
results = []
models = []
for i, pair in enumerate(pairs):
    try:
        # Select two columns at a time
        data = df[list(pair)]
        
        # Create the VAR model
        model = VAR(data)
        
        # Fit the model with optimal lag order selected by AIC
        result = model.fit(maxlags=5, ic='aic')
        
        # Append the model and result to the lists
        models.append(model)
        results.append(result)
        
        # Save the result summary to a text file
        with open(f'VAR results/first_run/result_{i}.txt', 'w') as file:
            file.write(str(result.summary()))
            
    except Exception as e:
        print(f"An error occurred when processing columns {pair}: {e}")
coe_set = set([tuple(sorted(pair)) for pair, _ in coe])

for i, result in enumerate(results):
    var_pair = tuple(sorted(result.names))
    if var_pair in coe_set:
        mb.append(var_pair);print(f"The pair {var_pair} exists in coe at index {i} in the results list.")
# Go through the results list
for i, result in enumerate(results):
    # Get the pair of variables from the result names and sort them
    var_pair = tuple(sorted(result.names))
    
    # Check if the sorted pair exists in coe
    if var_pair in coe_set:
        print(f"The pair {var_pair} exists in coe at index {i} in the results list.")

# Perform differencing Just once and retest for stationarity
non_stationary_columns = [column for column in df.columns if column not in stationary_columns_p_value]
for column in non_stationary_columns:
    df[column] = df[column].diff()
    try:
        result = adfuller(df[column], regression='c')
        p_value = result[1]
        if p_value < significance_level:
            stationary_columns_p_value.append(column)
    except Exception as e:
        print(f"An error occurred when processing column {column}: {e}")
        continue

# Generate all possible pairs of stationary variables
pairs = list(combinations(stationary_columns_p_value, 2))

# Perform Vector autoregression on the stationary columns two by two
results = []
models = []
for i, pair in enumerate(pairs):
    try:
        # Select two columns at a time
        data = df[list(pair)]
        
        # Create the VAR model
        model = VAR(data)
        
        # Fit the model with optimal lag order selected by AIC
        result = model.fit(maxlags=5, ic='aic')
        
        # Append the model and result to the lists
        models.append(model)
        results.append(result)
        
        # Save the result summary to a text file
        with open(f'VAR results/first_run/result_{i}.txt', 'w') as file:
            file.write(str(result.summary()))
            
    except Exception as e:
        print(f"An error occurred when processing columns {pair}: {e}")

# Initialize an empty list to store the existing pairs
existing_pairs = []
# Go through the results list
for i, result in enumerate(results):
    # Get the pair of variables from the result names and sort them
    var_pair = tuple(sorted(result.names))
    
    # Check if the sorted pair exists in coe
    if var_pair in coe_set:
        # If the pair exists, append it to the existing_pairs list
        existing_pairs.append(var_pair)
        print(f"The pair {var_pair} exists in coe at index {i} in the results list.")

# One line code to perform VAR
# data=df[list(('behavior networks','pathfinding'))];model = VAR(data);lt = model.fit(maxlags=5, ic='aic');lt.summary()

# Assuming pairs_corr_1 is your list of pairs with correlation of 1
pairs_corr_1 = [(var1, var2) for var1, var2 in pairs if df[var1].corr(df[var2]) == 1]

# find pairs that Co_occured and have a correlation of 1
# Initialize an empty list to store the existing pairs
existing_pairs_corr_1 = []

# Go through the pairs_corr_1 list
for i, pair in enumerate(pairs_corr_1):
    # Get the pair of variables and sort them
    var_pair = tuple(sorted(pair))
    
    # Check if the sorted pair exists in coe
    if var_pair in coe_set:
        # If the pair exists, append it to the existing_pairs_corr_1 list
        existing_pairs_corr_1.append(var_pair)
        print(f"The pair {var_pair} exists in coe at index {i} in the pairs_corr_1 list.")

# Find pairs that have correlation of 1 but did not co_occured
# Initialize an empty list to store the non-existing pairs
non_existing_pairs_corr_1 = []

# Go through the pairs_corr_1 list
for i, pair in enumerate(pairs_corr_1):
    # Get the pair of variables and sort them
    var_pair = tuple(sorted(pair))
    
    # Check if the sorted pair does not exist in coe
    if var_pair not in coe_set:
        # If the pair does not exist, append it to the non_existing_pairs_corr_1 list
        non_existing_pairs_corr_1.append(var_pair)
        print(f"The pair {var_pair} has a correlation of 1 but does not exist in coe.")
        
# Find pairs that has a correlation of 1, did not co_occured, and does not have
# exactly the same data
# Initialize an empty list to store the unique non-existing pairs
unique_non_existing_pairs_corr_1 = []

# Go through the non_existing_pairs_corr_1 list
for pair in non_existing_pairs_corr_1:
    # Get the pair of variables
    var1, var2 = pair
    
    # Check if the data in the two variables is not exactly the same
    if not df[var1].equals(df[var2]):
        # If the data is not the same, append the pair to the unique_non_existing_pairs_corr_1 list
        unique_non_existing_pairs_corr_1.append(pair)
        print(f"The pair {pair} has a correlation of 1, does not exist in coe, and does not have exactly the same data.")

# Find identical pairs with correlation of 1 not existing in keywords co_occurrence
# Initialize an empty list to store the identical non-existing pairs
identical_non_existing_pairs_corr_1 = []

# Go through the non_existing_pairs_corr_1 list
for pair in non_existing_pairs_corr_1:
    # Get the pair of variables
    var1, var2 = pair
    
    # Check if the data in the two variables is exactly the same
    if df[var1].equals(df[var2]):
        # If the data is the same, append the pair to the identical_non_existing_pairs_corr_1 list
        identical_non_existing_pairs_corr_1.append(pair)
        print(f"The pair {pair} has a correlation of 1, does not exist in coe, and has exactly the same data.")

# Calculate Variance Inflation Factor for each variable
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

# Print the VIF results
print(vif_data)

