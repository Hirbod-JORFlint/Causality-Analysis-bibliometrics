import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

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
        
#loading the Dset
time_df = pd.read_csv('diva_all.csv')

# Convert 'PublicationDate' to datetime and extract the year
#time_df['Year'] = time_df['Year'].astype(int)

# Convert 'Keywords' to lowercase
time_df['Keywords'] = time_df['Keywords'].str.lower()

# Initialize a new dataframe with 'Year' as the index
df_new = pd.DataFrame(index=time_df['Year'].unique().astype(int))

#[(('sweden',), 826), (('gender',), 665), (('sverige',), 617), (('heart failure',), 457), (('children',), 405), (('system identification',), 367), (('optimization',), 322), (('education',), 314), (('depression',), 313), (('identification',), 311), (('quality of life',), 303), (('epidemiology',), 295), (('simulation',), 265), (('covid-19',), 258), (('implementation',), 237), (('inflammation',), 233), (('estimation',), 227), (('communication',), 226), (('migration',), 216), (('learning',), 214)]
#items in above tuples are : ['sweden', 'genericpath', 'sverige', 'heart failure', 'children', 'system identification', 'optimization', 'education', 'depression', 'identification', 'quality of life', 'epidemiology', 'simulation', 'covid-19', 'implementation', 'inflammation', 'estimation', 'communication', 'migration', 'learning']

#[(('anxiety', 'depression'), 104), (('sverige', 'sweden'), 81), (('gender', 'genus'), 59), (('heart failure', 'self-care'), 50), (('gender', 'sexuality'), 47), (('health priorities', 'prioritering inom sjukvården'), 43), (('historia', 'sverige'), 41), (('cybernetik informationsteori', 'maskinelement servomekanismer automation'), 41), (('heart failure', 'mortality'), 40), (('children', 'type 1 diabetes'), 40), (('heart failure', 'prognosis'), 40), (('reliability', 'validity'), 39), (('gender', 'masculinity'), 37), (('bullying', 'bystander'), 37), (('bullying', 'moral disengagement'), 37), (('adolescents', 'children'), 35), (('modeling', 'simulation'), 34), (('genus', 'jämställdhet'), 34), (('stability', 'summation-by-parts'), 34), (('cortisol', 'stress'), 33)]
#items in above tuples are : ['sweden', 'gender', 'sverige', 'heart failure', 'children', 'system identification', 'optimization', 'education', 'depression', 'identification', 'quality of life', 'epidemiology', 'simulation', 'covid-19', 'implementation', 'inflammation', 'estimation', 'communication', 'migration', 'learning', 'anxiety', 'depression', 'sverige', 'sweden', 'gender', 'genus', 'heart failure', 'self-care', 'gender', 'sexuality', 'health priorities', 'prioritering inom sjukvården', 'historia', 'sverige', 'cybernetik informationsteori', 'maskinelement servomekanismer automation', 'heart failure', 'mortality', 'children', 'type 1 diabetes', 'heart failure', 'prognosis', 'reliability', 'validity', 'gender', 'masculinity', 'bullying', 'bystander', 'bullying', 'moral disengagement', 'adolescents', 'children', 'modeling', 'simulation', 'genus', 'jämställdhet', 'stability', 'summation-by-parts', 'cortisol', 'stress']

#(('recognition of prior learning', 'validation', 'validering'), 19), (('genus', 'jämställdhet', 'manlighet'), 17), (('health priorities', 'prioritering inom sjukvården', 'sverige'), 17), (('crohns disease', 'inflammatory bowel disease', 'ulcerative colitis'), 16), (('identitet', 'modernitet', 'ungdomskultur'), 14), (('faderskap', 'genus', 'manlighet'), 14), (('faderskap', 'genus', 'jämställdhet'), 14), (('metric space', 'nonlinear', 'p-harmonic'), 14), (('depression', 'meta-analysis', 'psychotherapy'), 14), (('automatisk styrning', 'reglerteori', 'systems'), 13), (('reglerteori', 'systems', 'systemteori'), 13), (('automatisk styrning', 'reglerteori', 'systemteori'), 13), (('automatisk styrning', 'systems', 'systemteori'), 13), (('forskning', 'identitet', 'ungdomskultur'), 13), (('female', 'humans', 'male'), 13), (('adaptive filtering', 'biological vision', 'vision structure'), 13), (('adaptive filtering', 'biological vision', 'seende datorer'), 13), (('adaptive filtering', 'biological vision', 'computer vision'), 13), (('adaptive filtering', 'biological vision', 'databehandling bildbehandling'), 13), (('adaptive filtering', 'seende datorer', 'vision structure'), 13)
#items in above tuples are : ['recognition of prior learning', 'validation', 'validering', 'genus', 'jämställdhet', 'manlighet', 'health priorities', 'prioritering inom sjukvården', 'sverige', 'crohns disease', 'inflammatory bowel disease', 'ulcerative colitis', 'identitet', 'modernitet', 'ungdomskultur', 'faderskap', 'genus', 'manlighet', 'faderskap', 'genus', 'jämställdhet', 'metric space', 'nonlinear', 'p-harmonic', 'depression', 'meta-analysis', 'psychotherapy', 'automatisk styrning', 'reglerteori', 'systems', 'reglerteori', 'systems', 'systemteori', 'automatisk styrning', 'reglerteori', 'systemteori', 'automatisk styrning', 'systems', 'systemteori', 'forskning', 'identitet', 'ungdomskultur', 'female', 'humans', 'male', 'adaptive filtering', 'biological vision', 'vision structure', 'adaptive filtering', 'biological vision', 'seende datorer', 'adaptive filtering', 'biological vision', 'computer vision', 'adaptive filtering', 'biological vision', 'databehandling bildbehandling', 'adaptive filtering', 'seende datorer', 'vision structure']


# Count the occurrences of list items in the 'Keywords' column for each year['quality of life', 'stress', 'validation', 'systems', 'reliability', 'sverige', 'prognosis', 'humans', 'computer vision', 'children', 'inflammatory bowel disease', 'faderskap', 'optimization', 'adaptive filtering', 'validering', 'vision structure', 'gender', 'psychotherapy', 'cybernetik informationsteori', 'male', 'genericpath', 'seende datorer', 'sexuality', 'modeling', 'jämställdhet', 'genus', 'cortisol', 'meta-analysis', 'ulcerative colitis', 'sweden', 'masculinity', 'systemteori', 'identitet', 'maskinelement servomekanismer automation', 'learning', 'reglerteori', 'biological vision', 'prioritering inom sjukvården', 'epidemiology', 'mortality', 'manlighet', 'heart failure', 'p-harmonic', 'communication', 'anxiety', 'historia', 'crohns disease', 'implementation', 'validity', 'automatisk styrning', 'female', 'system identification', 'recognition of prior learning', 'education', 'databehandling bildbehandling', 'nonlinear', 'stability', 'ungdomskultur', 'moral disengagement', 'depression', 'identification', 'inflammation', 'self-care', 'forskning', 'migration', 'bullying', 'bystander', 'simulation', 'type 1 diabetes', 'adolescents', 'estimation', 'summation-by-parts', 'modernitet', 'metric space', 'covid-19', 'health priorities']
keywords = ['quality of life', 'stress', 'validation', 'systems', 'reliability', 'sverige', 'prognosis', 'humans', 'computer vision', 'children', 'inflammatory bowel disease', 'faderskap', 'optimization', 'adaptive filtering', 'validering', 'vision structure', 'gender', 'psychotherapy', 'cybernetik informationsteori', 'male', 'genericpath', 'seende datorer', 'sexuality', 'modeling', 'jämställdhet', 'genus', 'cortisol', 'meta-analysis', 'ulcerative colitis', 'sweden', 'masculinity', 'systemteori', 'identitet', 'maskinelement servomekanismer automation', 'learning', 'reglerteori', 'biological vision', 'prioritering inom sjukvården', 'epidemiology', 'mortality', 'manlighet', 'heart failure', 'p-harmonic', 'communication', 'anxiety', 'historia', 'crohns disease', 'implementation', 'validity', 'automatisk styrning', 'female', 'system identification', 'recognition of prior learning', 'education', 'databehandling bildbehandling', 'nonlinear', 'stability', 'ungdomskultur', 'moral disengagement', 'depression', 'identification', 'inflammation', 'self-care', 'forskning', 'migration', 'bullying', 'bystander', 'simulation', 'type 1 diabetes', 'adolescents', 'estimation', 'summation-by-parts', 'modernitet', 'metric space', 'covid-19', 'health priorities']

df_new = pd.DataFrame()

for keyword in keywords:
    df_new[keyword] = time_df[time_df['Keywords'].str.contains(keyword, na=False)].groupby('Year').size()

# Fill NaN values with 0
for keyword in keywords:
    df_new[keyword].fillna(0, inplace=True)

# sort the dataset by year
df_new.sort_index(inplace=True)

# Save the new dataframe to a csv file
df_new.to_csv('time_series.csv')  # replace with your desired output filename

print("The new dataframe was successfully created and saved as new_file.csv.")
print(df_new)

# Set 'Year' as the index if int is not already.

#df_new.set_index('Year', inplace=True)

# Perform ADF test on each column
for column in df_new.columns:
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
