import pandas as pd
from statsmodels.tsa.stattools import adfuller
from itertools import combinations

# Load the dataset
df = pd.read_csv("time_series.csv")

# Drop the 'Year' column
df.drop(columns=['Year'], inplace=True)

# Drop constant columns
df = df.loc[:, df.apply(lambda x: len(x.unique()) > 1)]




