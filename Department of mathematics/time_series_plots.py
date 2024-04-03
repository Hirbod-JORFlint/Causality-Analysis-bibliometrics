import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv('time_series_compound.csv')

# Create a directory to save the figures if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Loop over each column
for column in df.columns:
    # Skip the 'Year' column
    if column == 'Year':
        continue

    # Draw a line plot for the current column
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'].values, df[column].values)  # Convert Series to numpy array
    plt.title(column)
    plt.xlabel('Year')
    plt.ylabel('Value')

    # Save the figure with the column name as the filename in the 'figures' directory
    plt.savefig(f'figures/{column}.png')

    # Close the figure to free up memory
    plt.close()

print("Plots saved successfully.")
