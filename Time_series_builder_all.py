import pandas as pd
from itertools import combinations
from collections import Counter
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os


# Load the dataset
time_df = pd.read_csv('final_df_processed.csv')

# Lowercase the keywords and split them into a list
time_df['Keywords'] = time_df['Keywords'].str.split(';').apply(lambda x: [word.strip().lower() for word in x])

# Add a new column 'keyword_count' that counts the number of keywords in each list
time_df['keyword_count'] = time_df['Keywords'].apply(len)

# Print the statistics of the 'keyword_count' column
print(time_df['keyword_count'].describe())

# Get the unique 'Year' values
years = sorted(time_df['Year'].unique())

# Explode the 'Keywords' column into multiple rows
time_df_exploded = time_df.explode('Keywords')

# Count the occurrences of each keyword for each year
df_keyword_counts = time_df_exploded.groupby(['Year', 'Keywords']).size().unstack(fill_value=0)

# Save the new dataframe to a csv file
df_keyword_counts.to_csv('time_series.csv')  # replace with your desired output filename

print(df_keyword_counts.head())

# for the big data
df_sum = df_keyword_counts.sum()
fre_df_keywords = list(zip(df_sum.index, df_sum.values))
fre_df_keywords.sort(key=lambda x: x[1], reverse=True)
print(fre_df_keywords[:40])


# Filter the DataFrame to include only the keywords that have a count of 0 for the years before 2019
df_before_2019 = df_keyword_counts[df_keyword_counts.index < 2019]
keywords_before_2019 = df_before_2019.columns[(df_before_2019 == 0).all()]

# Filter the DataFrame to include only the keywords that have a count greater than 0 for the years from 2019 onwards
df_from_2019 = df_keyword_counts[df_keyword_counts.index >= 2019]
keywords_from_2019 = df_from_2019.columns[(df_from_2019 > 0).any()]

# Get the intersection of the two sets of keywords
filtered_keywords = keywords_before_2019.intersection(keywords_from_2019)

# Filter the original DataFrame to include only these keywords
filtered_df = df_keyword_counts[filtered_keywords]
# Remove the year index before 2019
filtered_df = filtered_df[filtered_df.index >= 2019]

# Save the filtered DataFrame to a csv file

filtered_df.to_csv('filtered_time_series.csv')

print(filtered_df.head())

# Calculate the sum of each keyword's counts across all years
filt_keyword_sums = filtered_df.sum()

# Create a list of tuples, where each tuple contains a keyword and its sum
fre_keywords = list(zip(filt_keyword_sums.index, filt_keyword_sums.values))

# Sort the list of tuples by the sum in descending order
fre_keywords.sort(key=lambda x: x[1], reverse=True)

print(fre_keywords[:40])




def save_trend_analysis_plots(df, top_keywords, output_folder='trend_analysis'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Group the keywords by two
    keyword_groups = [top_keywords[i:i+2] for i in range(0, len(top_keywords), 2)]

    # Loop over each group of keywords
    for group in keyword_groups:
        # Create a new figure with 1x2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Loop over each keyword in the group
        for i, keyword in enumerate(group):
            # Skip if the keyword is not in the dataframe
            if keyword not in df.columns:
                continue

            # Plot the trend for the current keyword
            axs[i].plot(df[keyword], marker='o')
            axs[i].set_title(f'Trend Analysis for {keyword}')

            # Calculate the regression line
            x = np.array(df.index)
            y = df[keyword].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Calculate the x value where y=0 using the equation of the line
            x_at_y_equals_zero = -intercept / slope if slope != 0 else 0

            # Create a new x array starting from x_at_y_equals_zero to the end
            x_new = np.linspace(x_at_y_equals_zero, x[-1], num=1000)

            # Plot the regression line starting from y=0
            axs[i].plot(x_new, intercept + slope * x_new, 'r', label='fitted line')

        # Save the figure to the output folder
        plt.savefig(f'{output_folder}/{"_".join(group)}.png')

        # Close the figure to free up memory
        plt.close(fig)



def calculate_keyword_trends(top_keywords_df, top_keywords):
    """
    Calculate trends for each keyword using linear regression.

    Parameters:
        top_keywords_df (DataFrame): DataFrame containing keyword frequencies over time.
        top_keywords (list): List of top keywords to analyze.

    Returns:
        dict: Dictionary containing trend information for each keyword.
    """
    trends = {}

    # Loop over each keyword
    for keyword in top_keywords:
        # Skip if the keyword is not in the dataframe
        if keyword not in top_keywords_df.columns:
            continue

        # Calculate the regression line
        x = np.array(top_keywords_df.index)
        y = top_keywords_df[keyword].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Store the trend information in the dictionary
        trends[keyword] = {
            'slope': slope,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        }

    return trends




# Example usage
top_keywords = [keyword for keyword, _ in fre_df_keywords[:40]]
save_trend_analysis_plots(df_keyword_counts, top_keywords)
trends = calculate_keyword_trends(df_keyword_counts, top_keywords)

# Print results
for keyword, trend_info in trends.items():
    print(f"Keyword: {keyword}")
    print(f"Slope: {trend_info['slope']}")
    print(f"R-squared: {trend_info['r_value']**2}")
    print(f"P-value: {trend_info['p_value']}")
    print(f"Standard Error: {trend_info['std_err']}")
    print()


#top_keywords = [keyword for keyword, _ in fre_keywords[:40]]
#save_trend_analysis_plots(filt_keyword_sums, top_keywords, output_folder='trend_analysis/recent')
#trends = calculate_keyword_trends(filt_keyword_sums, top_keywords)





