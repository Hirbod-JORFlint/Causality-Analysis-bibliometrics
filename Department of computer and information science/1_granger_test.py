from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

# Function for performing Augmented Dickey-Fuller
def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    # Return the conclusion based on the p-value
    return result[1] < 0.05

# Load the data
df_new = pd.read_csv('time_series.csv')

# Lists to store the names of stationary and non-stationary columns
stationary_columns = []
non_stationary_columns = []

# Perform ADF test on each column
for column in df_new.columns:
    # Skip the column if it has constant values
    if column == 'Year' or df_new[column].nunique() == 1:
        print(f"\nSkipping ADF test on column: {column} because it has constant values or it is Year.")
        continue

    print(f"\nPerforming ADF test on column: {column}")
    is_stationary = perform_adf_test(df_new[column])

    # Add the column to the appropriate list
    if is_stationary:
        stationary_columns.append(column)
    else:
        non_stationary_columns.append(column)

# Print the lists of stationary and non-stationary columns
print("\nStationary columns:", stationary_columns)
print("Non-stationary columns:", non_stationary_columns)

# Transform non-stationary columns using differencing
for column in non_stationary_columns:
    df_new[column] = df_new[column].diff()

# Drop the first row after differencing
df_new = df_new.dropna()

# Print the transformed dataframe
print("\nTransformed dataframe:")
print(df_new)

#print(0+'r')
# Add a small noise to the data
df_new = df_new + np.random.normal(0, 0.001, size=df_new.shape)
df_new = df_new.drop(columns=['Year'])
# List to store the pairs of columns that have significant Granger causality
significant_pairs = []

# Perform the Granger causality test for each pair of columns
for keyword1 in df_new.columns:
    for keyword2 in df_new.columns:
        if keyword1 != keyword2:  # Exclude self-referential tests
            # Check if columns are not constant
            if df_new[keyword1].nunique() > 1 and df_new[keyword2].nunique() > 1:
                # Conduct Granger causality test
                test_result = grangercausalitytests(df_new[[keyword1, keyword2]], maxlag=3)
                # Check the results (change the significance level if needed)
                if test_result[1][0]['ssr_ftest'][1] < 0.05:  # Check p-value for significance
                    print(f"Granger causality test between '{keyword1}' and '{keyword2}' is significant.")
                    # Add the pair of columns to the list
                    significant_pairs.append((keyword1, keyword2))
                else:
                    pass
                    #print(f"Granger causality test between '{keyword1}' and '{keyword2}' is not significant.")
            else:
                print(f"Cannot perform Granger causality test between '{keyword1}' and '{keyword2}' as one or both are constant.")

# Print the list of significant pairs
print("\nSignificant pairs:", significant_pairs)
