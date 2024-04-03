import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
df = pd.read_csv('time_series.csv')

# Drop the 'Year' column
df = df.drop(columns=['Year'])

# Drop constant columns
df = df.loc[:, (df != df.iloc[0]).any()]

# Perform stationary testing and store the stationary variables
stationary_columns = []
for column in df.columns:
    result = adfuller(df[column])
    if result[1] < 0.05:  # if p-value < 0.05, the series is stationary
        stationary_columns.append(column)

# Perform Vector autoregression on the stationary columns three by three
results = []
for i in range(0, len(stationary_columns), 3):
    try:
        model = VAR(df[stationary_columns[i:i+3]])
        model_fit = model.fit()
        results.append(model_fit.summary())
    except np.linalg.LinAlgError:
        print(f"Skipping VAR for columns {stationary_columns[i:i+3]} due to LinAlgError")

# Save the results in a new csv file
with open('var_results.csv', 'w') as f:
    for result in results:
        f.write(str(result))

# Calculate Variance Inflation Factor for each variable
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

# Print the VIF results
print(vif_data)

