import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_table('Train_call.txt')  # Shows the data with samples as rows, so need to transpose
data = data.T  # Transposes data, so that samples are now rows.
data_target = pd.read_table('Train_clinical.txt')  # Gives sample with associated subgroup

# Extract predictor and target data
target = data_target.loc[:,
         "Subgroup"]  # Isolates the subgroups from samples. We need to convert the subgroups into 0, 1, 2
new_data_unlabeled = data.iloc[4:, :]  # This is the complete cleaned up dataset

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_data_unlabeled, target, test_size=0.2,
                                                    random_state=42)

### INITIAL FEATURE SELECTION ###
# Combining X_train and y_train again in one dataframe
y_train_df = pd.DataFrame(y_train)
train_data = X_train
train_data['Subgroup'] = list(y_train_df.iloc[:,0])

# Perform Chi-Squared test for each feature (column)
chi2_results = []
target_variable = 'Subgroup'

for feature in train_data.columns:
    if feature != target_variable:  # Check if the feature is categorical
        contingency_table = pd.crosstab(train_data[feature], train_data[target_variable])
        results = chi2_contingency(contingency_table)
        chi2_results.append({'Feature': feature, 'Chi-Squared': results[0], 'p-value': results[1]})

df_chi = pd.DataFrame(chi2_results) # Save the results into a dataframe

# features = df_chi[df_chi['p-value']<0.01]
# len(features)
# --> 152

# Applying FDR control with Benjamini-Hochberg procedure
df_chi = df_chi.sort_values(by=['p-value'])  # arranging the results by ascending p-values
q = 0.1    # setting FDR threshold, allowing for 10% FPs
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
# --> 10 

features = pd.DataFrame(significant_features['Feature'])
features.to_csv('output_feature_selection.txt', index= False)
