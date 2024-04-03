import pandas as pd

def extract_keywords_and_save(input_csv, output_csv):
    # Read the input CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Extract the 'Keywords' column
    keywords_column = df['Keywords']

    # Save the extracted column to a new CSV file
    keywords_column.to_csv(output_csv, index=False, header=['Keywords'])

# Example usage:
input_csv_file = "updated_file.csv"
output_csv_file = "keywords_column.csv"
extract_keywords_and_save(input_csv_file, output_csv_file)
