import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


# Final features and hps:
hp_rf = {'criterion': 'gini', 'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 40, 'random_state': 42}
hp_dt = {'criterion': 'gini', 'max_depth': None, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'random_state': 42}
hp_reg_lr = {'C': 1291.5496650148827, 'max_iter': 2000, 'penalty': 'l2', 'random_state': 42, 'solver': 'lbfgs'}
feats = [12, 15, 17, 32, 308, 354, 474, 480, 484, 485, 487, 489, 499, 554, 555, 558, 594, 610, 612, 615, 623, 674, 679, 693, 700, 718, 724, 725, 727, 729, 733, 743, 758, 849, 874, 998, 1001, 1105, 1281, 1295, 1383, 1423, 1598, 1606, 1635, 1636, 1638, 1641, 1645, 1650, 1651, 1655, 1660, 1679, 1681, 1682, 1684, 1690, 1701, 1862, 1877, 1879, 1896, 1902, 1910, 1913, 1950, 1963, 1973, 2047, 2056, 2154, 2168, 2183, 2184, 2185, 2196, 2205, 2213, 2220, 2221, 2223, 2226, 2241, 2281, 2285, 2293, 2379, 2661, 2663, 2709, 2722, 2733, 2751, 2760, 2763, 2765, 2774, 2831, 2833]

# Data preprocessing

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

# Restrict dataset to optimal features
X_train = new_data_unlabeled.iloc[:, feats]
y_train = target.astype(int)

# Train optimal model
rf_classifier = RandomForestClassifier(**hp_rf)
rf_classifier.fit(X_train, y_train)
joblib.dump(rf_classifier, filename ='model.pkl')