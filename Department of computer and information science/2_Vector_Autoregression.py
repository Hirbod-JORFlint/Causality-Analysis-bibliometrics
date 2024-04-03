import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Load the dataset
df = pd.read_csv('time_series.csv')

# Define the list of tuples
tuples = [(('visual analytics', 'visualization'), 16), (('haptics', 'visualization'), 10), (('human-centered computing', 'visualization'), 10)]

# Define a function to perform the Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series)
    return result[1] <= 0.05  # p-value

# Iterate over each tuple in the list
for tuple in tuples:
    # Extract the column names from the tuple
    col1, col2 = tuple[0]

    # Check if the columns exist in the dataframe
    if col1 in df.columns and col2 in df.columns:
        # Check if the columns are stationary
        if adf_test(df[col1]) and adf_test(df[col2]):
            # Perform VAR
            model = VAR(df[[col1, col2]])
            results = model.fit(maxlags=5, ic='aic')

            # Print the VAR results
            print(f"VAR Results for {col1} and {col2}:")
            print(results.summary())
        else:
            print(f"Either {col1} or {col2} is not stationary. VAR cannot be performed.")
    else:
        print(f"Either {col1} or {col2} does not exist in the dataframe.")

