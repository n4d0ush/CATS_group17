import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot

# Load data
data = pd.read_table('Train_call.txt')  # Shows the data with samples as rows, so need to transpose
data = data.T  # Transposes data, so that samples are now rows.
data_target = pd.read_table('Train_clinical.txt')  # Gives sample with associated subgroup

# Extract predictor and target data
target = data_target.loc[:,
         "Subgroup"]  # Isolates the subgroups from samples. We need to convert the subgroups into 0, 1, 2
new_data_unlabeled = data.iloc[4:, :]  # This is the complete cleaned up dataset

# We also need to convert the subgroups into 0, 1, 2
for i in range(len(target)):
    if target[i] == "HER2+":
        target[i] = 0
    elif target[i] == "HR+":
        target[i] = 1
    elif target[i] == "Triple Neg":
        target[i] = 2

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_data_unlabeled, target, test_size=0.2,
                                                    random_state=42)
y_train = y_train.astype(int)  # For some reason the numbers are read as strings, so convert to integers

# The values have to be non-negative, so shifting them all
X_train_shifted = X_train - X_train.min() + 1  # Shift all values to be positive
X_test_shifted = X_test - X_test.min() + 1  # Shift all values to be positive


# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    oe.fit(X_test)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc

# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X_train_shifted, y_train)
X_train_fs = fs.transform(X_train_shifted)
X_test_fs = fs.transform(X_test_shifted)

chitest_results = pd.DataFrame({'scores': fs.scores_, 'pvalues': fs.pvalues_})
rslt_df = chitest_results[chitest_results['pvalues']< 0.4]
rslt_df.shape

for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

# for threshold p < 0.2, 17 features left...
# for threshold p < 0.4, 178 features left...

# --------------- When not splitting in advance ----------------------
# Load data
data = pd.read_table('Train_call.txt')  # Shows the data with samples as rows, so need to transpose
data = data.T  # Transposes data, so that samples are now rows.
data_target = pd.read_table('Train_clinical.txt')  # Gives sample with associated subgroup

# Extract predictor and target data
target = data_target.loc[:,
         "Subgroup"]  # Isolates the subgroups from samples. We need to convert the subgroups into 0, 1, 2
new_data_unlabeled = data.iloc[4:, :]  # This is the complete cleaned up dataset

# prepare input data
def prepare_inputs(X):
    oe = OrdinalEncoder()
    oe.fit(X)
    X_enc = oe.transform(X)
    return X_enc

X_all = prepare_inputs(new_data_unlabeled)
X_all_shifted = X_all - X_all.min() + 1  # Shift all values to be positive
# prepare output data
def prepare_targets(y_train):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    return y_train_enc
y_all = prepare_targets(target)

fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X_all_shifted, y_all)
X_all_fs = fs.transform(X_all_shifted)

# adjusted_p_values = fs.pvalues_ * X_all_shifted.shape[1]
# but then none of the features remains :(

chitest_results = pd.DataFrame({'scores': fs.scores_, 'pvalues': fs.pvalues_})
rslt_df = chitest_results[chitest_results['pvalues']< 0.2]
rslt_df.shape

# for p < 0.2, 57 features left...
# for p < 0.4, 320 features left...
for i in range(len(fs.pvalues_)):
 print('Feature %d: %f' % (i, fs.pvalues_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.pvalues_))], fs.pvalues_)
pyplot.show()
