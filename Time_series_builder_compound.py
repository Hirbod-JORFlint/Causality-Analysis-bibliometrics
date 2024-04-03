import pandas as pd
import numpy as np

"""
This script creates a time series for each keyword in a provided list by counting its occurrence in the 'Keywords' column of a dataset for each year, even if the keyword is part of a larger string.

The script performs the following steps:
1. Load the dataset.
2. Lowercase the keywords.
3. Split the 'Keywords' column into a list of keywords.
4. Get the unique 'Year' values.
5. Initialize a dictionary to store the count of each keyword for each year.
6. Go through each row in the dataframe.
7. If 'Keywords' is not missing and the keyword is in the list of keywords to check, increment the count.
8. Convert the dictionary to a dataframe.
9. Fill NaN values with 0.
10. Save the dataframe to a new CSV file.
"""

# Load the dataset
time_df = pd.read_csv('df_processed.csv')

# Lowercase the keywords
time_df['Keywords'] = time_df['Keywords'].str.lower()


# Get the unique 'Year' values
years = sorted(time_df['Year'].unique())

#[(('sweden',), 891), (('gender',), 690), (('sverige',), 609), (('heart failure',), 557), (('children',), 450), (('system identification',), 372), (('epidemiology',), 362), (('quality of life',), 344), (('depression',), 342), (('education',), 333), (('optimization',), 332), (('identification',), 313), (('simulation',), 277), (('covid-19',), 267), (('inflammation',), 258), (('implementation',), 244), (('communication',), 242), (('estimation',), 232), (('stability',), 232), (('mortality',), 228), (('prognosis',), 228), (('learning',), 224), (('migration',), 223), (('apoptosis',), 218), (('energy efficiency',), 216), (('type 1 diabetes',), 211), (('genus',), 210), (('sustainability',), 210), (('rehabilitation',), 207), (('internet',), 201), (('pain',), 198), (('innovation',), 195), (('pregnancy',), 192), (('visualization',), 192), (('dementia',), 192), (('parameter estimation',), 187), (('machine learning',), 187), (('stress',), 186), (('breast cancer',), 185), (('adolescents',), 185), (('elderly',), 183), (('nursing',), 181), (('diagnosis',), 180), (('health',), 179), (('anxiety',), 179), (('assessment',), 174), (('ethics',), 173), (('bullying',), 168), (('qualitative research',), 166), (('photoluminescence',), 163)]
#items in above tuples are : ['sweden', 'gender', 'sverige', 'heart failure', 'children', 'system identification', 'epidemiology', 'quality of life', 'depression', 'education', 'optimization', 'identification', 'simulation', 'covid-19', 'inflammation', 'implementation', 'communication', 'estimation', 'stability', 'mortality', 'prognosis', 'learning', 'migration', 'apoptosis', 'energy efficiency', 'type 1 diabetes', 'genus', 'sustainability', 'rehabilitation', 'internet', 'pain', 'innovation', 'pregnancy', 'visualization', 'dementia', 'parameter estimation', 'machine learning', 'stress', 'breast cancer', 'adolescents', 'elderly', 'nursing', 'diagnosis', 'health', 'anxiety', 'assessment', 'ethics', 'bullying', 'qualitative research', 'photoluminescence']

#[(('anxiety', 'depression'), 104), (('sverige', 'sweden'), 81), (('gender', 'genus'), 59), (('heart failure', 'self-care'), 50), (('gender', 'sexuality'), 47), (('health priorities', 'prioritering inom sjukvården'), 43), (('historia', 'sverige'), 41), (('cybernetik informationsteori', 'maskinelement servomekanismer automation'), 41), (('heart failure', 'mortality'), 40), (('children', 'type 1 diabetes'), 40), (('heart failure', 'prognosis'), 40), (('reliability', 'validity'), 39), (('gender', 'masculinity'), 37), (('bullying', 'bystander'), 37), (('bullying', 'moral disengagement'), 37), (('adolescents', 'children'), 35), (('modeling', 'simulation'), 34), (('genus', 'jämställdhet'), 34), (('stability', 'summation-by-parts'), 34), (('cortisol', 'stress'), 33)]
#items in above tuples are : ['sweden', 'gender', 'sverige', 'heart failure', 'children', 'system identification', 'optimization', 'education', 'depression', 'identification', 'quality of life', 'epidemiology', 'simulation', 'covid-19', 'implementation', 'inflammation', 'estimation', 'communication', 'migration', 'learning', 'anxiety', 'depression', 'sverige', 'sweden', 'gender', 'genus', 'heart failure', 'self-care', 'gender', 'sexuality', 'health priorities', 'prioritering inom sjukvården', 'historia', 'sverige', 'cybernetik informationsteori', 'maskinelement servomekanismer automation', 'heart failure', 'mortality', 'children', 'type 1 diabetes', 'heart failure', 'prognosis', 'reliability', 'validity', 'gender', 'masculinity', 'bullying', 'bystander', 'bullying', 'moral disengagement', 'adolescents', 'children', 'modeling', 'simulation', 'genus', 'jämställdhet', 'stability', 'summation-by-parts', 'cortisol', 'stress']

#[(('recognition of prior learning', 'validation', 'validering'), 19), (('genus', 'jämställdhet', 'manlighet'), 17), (('health priorities', 'prioritering inom sjukvården', 'sverige'), 17), (('depression', 'meta-analysis', 'psychotherapy'), 17), (('apoptosis', 'lysosomes', 'oxidative stress'), 16), (('metric space', 'nonlinear', 'p-harmonic'), 16), (('crohns disease', 'inflammatory bowel disease', 'ulcerative colitis'), 16), (('lysosomes', 'mitochondria', 'oxidative stress'), 15), (('autophagy', 'lysosomes', 'oxidative stress'), 15), (('identitet', 'modernitet', 'ungdomskultur'), 14), (('faderskap', 'genus', 'jämställdhet'), 14), (('faderskap', 'genus', 'manlighet'), 14), (('automatisk styrning', 'reglerteori', 'systemteori'), 13), (('automatisk styrning', 'reglerteori', 'systems'), 13), (('automatisk styrning', 'systems', 'systemteori'), 13), (('reglerteori', 'systems', 'systemteori'), 13), (('forskning', 'identitet', 'ungdomskultur'), 13), (('female', 'humans', 'male'), 13), (('computer vision', 'seende datorer', 'vision structure'), 13), (('computer vision', 'databehandling bildbehandling', 'seende datorer'), 13), (('biological vision', 'computer vision', 'seende datorer'), 13), (('adaptive filtering', 'computer vision', 'seende datorer'), 13), (('computer vision', 'databehandling bildbehandling', 'vision structure'), 13), (('biological vision', 'computer vision', 'vision structure'), 13), (('adaptive filtering', 'computer vision', 'vision structure'), 13), (('biological vision', 'computer vision', 'databehandling bildbehandling'), 13), (('adaptive filtering', 'computer vision', 'databehandling bildbehandling'), 13), (('adaptive filtering', 'biological vision', 'computer vision'), 13), (('databehandling bildbehandling', 'seende datorer', 'vision structure'), 13), (('biological vision', 'seende datorer', 'vision structure'), 13), (('adaptive filtering', 'seende datorer', 'vision structure'), 13), (('biological vision', 'databehandling bildbehandling', 'seende datorer'), 13), (('adaptive filtering', 'databehandling bildbehandling', 'seende datorer'), 13), (('adaptive filtering', 'biological vision', 'seende datorer'), 13), (('biological vision', 'databehandling bildbehandling', 'vision structure'), 13), (('adaptive filtering', 'databehandling bildbehandling', 'vision structure'), 13), (('adaptive filtering', 'biological vision', 'vision structure'), 13), (('adaptive filtering', 'biological vision', 'databehandling bildbehandling'), 13), (('faderskap', 'jämställdhet', 'manlighet'), 13), (('aging', 'lysosomes', 'oxidative stress'), 13), (('dagböcker', 'politik', 'tage erlander'), 13), (('article', 'controlled study', 'human'), 13), (('doubling measure', 'metric space', 'poincare inequality'), 13), (('sverige', 'vuxenundervisning', 'vuxenutbildning'), 13), (('heart failure', 'patient education', 'self-care'), 12), (('collagenous colitis', 'lymphocytic colitis', 'microscopic colitis'), 12), (('science fiction', 'teknik', 'vetenskap'), 12), (('health priorities', 'prioritering inom sjukvården', 'sweden'), 12), (('modelica', 'modeling', 'simulation'), 12), (('nonlinear', 'p-harmonic', 'potential theory'), 12)]
#items in above tuples are : ['recognition of prior learning', 'validation', 'validering', 'genus', 'jämställdhet', 'manlighet', 'health priorities', 'prioritering inom sjukvården', 'sverige', 'depression', 'meta-analysis', 'psychotherapy', 'apoptosis', 'lysosomes', 'oxidative stress', 'metric space', 'nonlinear', 'p-harmonic', 'crohns disease', 'inflammatory bowel disease', 'ulcerative colitis', 'lysosomes', 'mitochondria', 'oxidative stress', 'autophagy', 'lysosomes', 'oxidative stress', 'identitet', 'modernitet', 'ungdomskultur', 'faderskap', 'genus', 'jämställdhet', 'faderskap', 'genus', 'manlighet', 'automatisk styrning', 'reglerteori', 'systemteori', 'automatisk styrning', 'reglerteori', 'systems', 'automatisk styrning', 'systems', 'systemteori', 'reglerteori', 'systems', 'systemteori', 'forskning', 'identitet', 'ungdomskultur', '

# Count the goccurrences of list items in the 'Keywords' column for each year['quality of life', 'stress', 'validation', 'systems', 'reliability', 'sverige', 'prognosis', 'humans', 'computer vision', 'children', 'inflammatory bowel disease', 'faderskap', 'optimization', 'adaptive filtering', 'validering', 'vision structure', 'gender', 'psychotherapy', 'cybernetik informationsteori', 'male', 'genericpath', 'seende datorer', 'sexuality', 'modeling', 'jämställdhet', 'genus', 'cortisol', 'meta-analysis', 'ulcerative colitis', 'sweden', 'masculinity', 'systemteori', 'identitet', 'maskinelement servomekanismer automation', 'learning', 'reglerteori', 'biological vision', 'prioritering inom sjukvården', 'epidemiology', 'mortality', 'manlighet', 'heart failure', 'p-harmonic', 'communication', 'anxiety', 'historia', 'crohns disease', 'implementation', 'validity', 'automatisk styrning', 'female', 'system identification', 'recognition of prior learning', 'education', 'databehandling bildbehandling', 'nonlinear', 'stability', 'ungdomskultur', 'moral disengagement', 'depression', 'identification', 'inflammation', 'self-care', 'forskning', 'migration', 'bullying', 'bystander', 'simulation', 'type 1 diabetes', 'adolescents', 'estimation', 'summation-by-parts', 'modernitet', 'metric space', 'covid-19', 'health priorities']
keyword1= ['sustainability', 'mortality', 'migration', 'heart failure', 'optimization', 'qualitative research', 'covid-19', 'communication', 'diagnosis', 'education', 'parameter estimation', 'sweden', 'epidemiology', 'estimation', 'visualization', 'sverige', 'rehabilitation', 'machine learning', 'identification', 'nursing', 'dementia', 'inflammation', 'depression', 'internet', 'stress', 'prognosis', 'type 1 diabetes', 'innovation', 'stability', 'learning', 'anxiety', 'energy efficiency', 'children', 'breast cancer', 'bullying', 'quality of life', 'ethics', 'health', 'pregnancy', 'adolescents', 'assessment', 'pain', 'photoluminescence', 'simulation', 'elderly', 'gender', 'system identification', 'implementation', 'apoptosis', 'genus']
keyword2= ['anxiety', 'oxidative stress', 'circular economy', 'cytokines', 'maskinelement servomekanismer automation', 'stress', 'inflammation', 'recognition of prior learning', 'vuxenutbildning', 'soccer', 'lysosomes', 'jämställdhet', 'historia', 'bystander', 'communication', 'depression', 'gender', 'modeling', 'epidemiology', 'sars-cov-2', 'ulcerative colitis', 'tamoxifen', 'inflammatory bowel disease', 'children', 'foucault', 'remanufacturing', 'prioritering inom sjukvården', 'demenssjuka', 'validity', 'breast cancer', 'masculinity', 'type 1 diabetes', 'governmentality', 'heart failure', 'etnicitet', 'validation', 'channel estimation', 'health priorities', 'summation-by-parts', 'massive mimo', 'adolescents', 'cybernetik informationsteori', 'prognosis', 'cortisol', 'signal processing', 'genus', 'manlighet', 'nonlinear systems', 'men', 'self-care', 'sexuality', 'demens', 'feminist theory', 'quality of life', 'parameter estimation', 'moral disengagement', 'ethnicity', 'intersectionality', 'apoptosis', 'reliability', 'colorectal cancer', 'sverige', 'simulation', 'adult education', 'stability', 'mortality', 'covid-19', 'mobbning', 'bullying', 'system identification', 'health', 'sweden', 'kommunikation']
keyword3= ['female', 'mitochondria', 'tage erlander', 'patient education', 'humans', 'dagböcker', 'biological vision', 'vuxenutbildning', 'modernitet', 'forskning', 'health priorities', 'sweden', 'modelica', 'modeling', 'potential theory', 'manlighet', 'poincare inequality', 'lymphocytic colitis', 'apoptosis', 'databehandling bildbehandling', 'jämställdhet', 'faderskap', 'aging', 'prioritering inom sjukvården', 'identitet', 'oxidative stress', 'controlled study', 'doubling measure', 'science fiction', 'vetenskap', 'politik', 'validation', 'seende datorer', 'adaptive filtering', 'ungdomskultur', 'inflammatory bowel disease', 'autophagy', 'meta-analysis', 'self-care', 'simulation', 'validering', 'sverige', 'crohns disease', 'microscopic colitis', 'p-harmonic', 'automatisk styrning', 'vuxenundervisning', 'heart failure', 'collagenous colitis', 'lysosomes', 'article', 'human', 'depression', 'male', 'systems', 'computer vision', 'systemteori', 'vision structure', 'teknik', 'recognition of prior learning', 'nonlinear', 'genus', 'ulcerative colitis', 'metric space', 'reglerteori', 'psychotherapy']

# create a list of all keywords(unique)
keywords = list(set(keyword1 + keyword2 + keyword3))

# Initialize a dictionary to store the count of each keyword for each year
keyword_counts = {year: {keyword: 0 for keyword in keywords} for year in years}

# Go through each row in the dataframe
for _, row in time_df.iterrows():
    year = row['Year']
    row_keywords = row['Keywords']
    
    # If 'Keywords' is not missing
    if row_keywords is not np.nan:
        # Go through each keyword
        for keyword in keywords:
            # If the keyword is in the row_keywords
            if keyword in row_keywords:
                # Increment the count
                keyword_counts[year][keyword] += 1

# Convert the dictionary to a dataframe
df_keyword_counts = pd.DataFrame(keyword_counts).transpose()

# Fill NaN values with 0
df_keyword_counts.fillna(0, inplace=True)

# Save the new dataframe to a csv file
df_keyword_counts.to_csv('time_series_compound.csv')  # replace with your desired output filename

print("The new dataframe was successfully created and saved as new_file.csv.")
print(df_keyword_counts.head())

# Set 'Year' as the index if int is not already.

#df_new.set_index('Year', inplace=True)

