from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import pandas as pd

df_new = pd.read_csv('time_series.csv')
# Function for performing Augmented Dickey-Fuller
def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    # Print the conclusion based on the p-value
    if result[1] < 0.05:
        print("Conclusion: The series is stationary")
    else:
        print("Conclusion: The series is not stationary")

# Perform ADF test on each column
for column in df_new.columns:
    # Skip the column if it has constant values
    if df_new[column].nunique() == 1:
        print(f"\nSkipping ADF test on column: {column} because it has constant values.")
        continue

    print(f"\nPerforming ADF test on column: {column}")
    perform_adf_test(df_new[column])

print(0+'r')

# Perform the Granger causality test for each pair of columns


for keyword1 in df_new.columns:
    for keyword2 in df_new.columns:
        if keyword1 != keyword2:  # Exclude self-referential tests
            # Check if columns are not constant
            if df_new[keyword1].nunique() > 1 and df_new[keyword2].nunique() > 1:
                # Conduct Granger causality test
                test_result = grangercausalitytests(df_new[[keyword1, keyword2]], maxlag=1)
                # Print results (change the significance level if needed)
                if test_result[1][0]['ssr_ftest'][1] < 0.05:  # Check p-value for significance
                    print(f"Granger causality test between '{keyword1}' and '{keyword2}' is significant.")
                #else:
                    #print(f"Granger causality test between '{keyword1}' and '{keyword2}' is not significant.")
            else:
                print(f"Cannot perform Granger causality test between '{keyword1}' and '{keyword2}' as one or both are constant.")
