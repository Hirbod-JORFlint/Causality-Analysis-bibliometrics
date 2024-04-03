import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv('time_series.csv')

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
    plt.plot(df['Year'].values, df[column].values, marker='o')  # Convert Series to numpy array
    plt.title(column)
    plt.xlabel('Year')
    plt.ylabel('Value')

    # Save the figure with the column name as the filename in the 'figures' directory
    plt.savefig(f'figures/{column}.png')

    # Close the figure to free up memory
    plt.close()

print("Plots saved successfully.")


VARS = [(('automatisk styrning', 'reglerteori', 'systems', 'systemteori'), 13), (('adaptive filtering', 'computer vision', 'databehandling bildbehandling', 'seende datorer'), 13), (('adaptive filtering', 'computer vision', 'databehandling bildbehandling', 'vision structure'), 13), (('adaptive filtering', 'biological vision', 'computer vision', 'databehandling bildbehandling'), 13), (('computer vision', 'databehandling bildbehandling', 'seende datorer', 'vision structure'), 13), (('biological vision', 'computer vision', 'databehandling bildbehandling', 'seende datorer'), 13), (('biological vision', 'computer vision', 'databehandling bildbehandling', 'vision structure'), 13), (('adaptive filtering', 'databehandling bildbehandling', 'seende datorer', 'vision structure'), 13), (('adaptive filtering', 'biological vision', 'databehandling bildbehandling', 'seende datorer'), 13), (('adaptive filtering', 'biological vision', 'databehandling bildbehandling', 'vision structure'), 13), (('biological vision', 'databehandling bildbehandling', 'seende datorer', 'vision structure'), 13), (('adaptive filtering', 'computer vision', 'seende datorer', 'vision structure'), 13), (('adaptive filtering', 'biological vision', 'computer vision', 'seende datorer'), 13), (('adaptive filtering', 'biological vision', 'computer vision', 'vision structure'), 13), (('biological vision', 'computer vision', 'seende datorer', 'vision structure'), 13), (('adaptive filtering', 'biological vision', 'seende datorer', 'vision structure'), 13), (('faderskap', 'genus', 'jämställdhet', 'manlighet'), 12), (('automatisk styrning', 'control theory -- dictionaries', 'systems', 'systemteori'), 11), (('control theory -- dictionaries', 'reglerteori', 'systems', 'systemteori'), 11), (('automatisk styrning', 'control theory -- dictionaries', 'reglerteori', 'systemteori'), 11), (('automatisk styrning', 'control theory -- dictionaries', 'reglerteori', 'systems'), 11), (('metric space', 'nonlinear', 'p-harmonic', 'potential theory'), 11), (('forskning', 'identitet', 'modernitet', 'ungdomskultur'), 10), (('dead-time', 'estimation', 'system identification', 'time-delay'), 10), (('adult', 'article', 'human', 'male'), 10), (('adult', 'article', 'controlled study', 'human'), 10), (('humanities', 'man', 'människans väsen', 'posthumanism'), 10), (('humanities', 'människans väsen', 'philosophy', 'posthumanism'), 10), (('humanities', 'man', 'människans väsen', 'philosophy'), 10), (('man', 'människans väsen', 'philosophy', 'posthumanism'), 10), (('humanities', 'man', 'philosophy', 'posthumanism'), 10), (('opartiskhet', 'rättssäkerhet', 'saklighet', 'tankefel'), 10), (('aging', 'lysosomes', 'mitochondria', 'oxidative stress'), 9), (('health priorities', 'prioritering inom sjukvården', 'sverige', 'sweden'), 9), (('gender', 'genus', 'sexualitet', 'sexuality'), 9), (('adult', 'controlled study', 'female', 'human'), 9), (('adult', 'article', 'female', 'human'), 9), (('adult', 'article', 'controlled study', 'female'), 9), (('adult', 'controlled study', 'human', 'male'), 9), (('adult', 'article', 'controlled study', 'male'), 9), (('article', 'controlled study', 'female', 'human'), 9), (('article', 'controlled study', 'human', 'male'), 9), (('adult education', 'folkbildning', 'non-formal education', 'vuxenutbildning'), 9), (('adult education', 'continuing education', 'non-formal education', 'vuxenutbildning'), 9), (('continuing education', 'folkbildning', 'non-formal education', 'vuxenutbildning'), 9), (('adult education', 'continuing education', 'folkbildning', 'non-formal education'), 9), (('continuing education', 'folkbildning', 'livslångt lärande', 'vuxenutbildning'), 9), (('adult education', 'continuing education', 'folkbildning', 'vuxenutbildning'), 9), (('bullying', 'mobbning', 'moral disengagement', 'moraliskt disengagemang'), 9), (('brister i utredningsmetodik', 'opartiskhet', 'saklighet', 'tankefel'), 9)]
def plot_time_series(file, variables):
    # Create the directory if it doesn't exist
    if not os.path.exists('triplets'):
        os.makedirs('triplets')

    # Load the dataset
    df = pd.read_csv(file)

    # Iterate over the variables
    for variable in variables:
        # Extract the variable names from the first inner tuple
        var_names = variable[0]

        # Create a subplot for each variable
        fig, axs = plt.subplots(len(var_names), figsize=(10, 5*len(var_names)))

        # If there's only one variable, axs is not a list
        if len(var_names) == 1:
            axs = [axs]

        # Plot each variable
        for i, var_name in enumerate(var_names):
            axs[i].plot(df['Year'].values, df[var_name].values, marker='o')
            axs[i].set_title(var_name)
            axs[i].set_xlabel('Year')

        # Save the plot to the 'triplets' directory
        plt.savefig(f'triplets/{var_names}.png', bbox_inches='tight')

        # Close the plot to free up memory
        plt.close(fig)

plot_time_series('time_series.csv', VARS)
