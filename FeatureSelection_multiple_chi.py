import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load data and merge the two dataframes to combine features with subgroup
data = pd.read_table('Train_call.txt')
new_data = data.iloc[:,4:] # remove first 4 columns
new_data = new_data.T # transpose to put features as columns

data_target = pd.read_table('Train_clinical.txt')  # Gives sample with associated subgroup
data_target = data_target.T
data_target.columns= data_target.iloc[0,:]   # column names should be the sample
data_target = data_target.iloc[1:]          # remove first row
data_target = data_target.T            # transpose again to get the samples as rows

new_data['Subgroup']= data_target.iloc[:,0]

# Perform Chi-Squared test for each feature (column)
chi2_results = []
target_variable = 'Subgroup'

for feature in new_data.columns:
    if feature != target_variable:  # Check if the feature is categorical
        contingency_table = pd.crosstab(new_data[feature], new_data[target_variable])
        results = chi2_contingency(contingency_table)
        chi2_results.append({'Feature': feature, 'Chi-Squared': results[0], 'p-value': results[1]})

df_chi = pd.DataFrame(chi2_results) # Save the results into a dataframe

# features = df_chi[df_chi['p-value']<0.01]
# len(features)
# --> 304

# Applying FDR control with Benjamini-Hochberg procedure
df_chi = df_chi.sort_values(by=['p-value'])  # arranging the results by ascending p-values
q = 0.05    # setting FDR threshold, allowing for 5% FPs
m = len(df_chi['p-value'])
critical_values = []
for i in range(1, m+1):
    critical_value = (i/m)*q
    critical_values.append(critical_value)
df_chi['critical_value']= critical_values

significant = []
for i in range(0, m):
    if df_chi.iloc[i,2] <= df_chi.iloc[i,3]:    # when p-value is smaller than or equal to critical value
        significant.append(True)
    else:
        significant.append(False)

df_chi['significant'] = significant

# Select the significant features
significant_features = df_chi[df_chi['significant']]
print(len(significant_features))
# --> 175