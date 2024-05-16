import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier  # Classifier - Random Forest
from sklearn.linear_model import LogisticRegression  # Classifier - Logistic Regression
from sklearn.tree import DecisionTreeClassifier  # Classifier - Decision Tree

from sklearn.metrics import accuracy_score  # Evaluation metric - Accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from yellowbrick import ROCAUC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay

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

# Now split data into train/test set with OPTIMAL features
new_data_unlabeled = new_data_unlabeled.iloc[:, feats]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_data_unlabeled, target, test_size=0.2,
                                                    random_state=42)
y_train = y_train.astype(int)  # For some reason the numbers are read as strings, so convert to integers

# Train optimal model
rf_classifier = RandomForestClassifier(**hp_rf)
dt_classifier = DecisionTreeClassifier(**hp_dt)
lr_reg_classifier = LogisticRegression(**hp_reg_lr)

rf_classifier.fit(X_train, y_train)
dt_classifier.fit(X_train, y_train)
lr_reg_classifier.fit(X_train, y_train)

# Test performance on test set
rf_y_predict = rf_classifier.predict(X_test)
dt_y_predict = dt_classifier.predict(X_test)
lr_reg_y_predict = lr_reg_classifier.predict(X_test)

# Compute accuracy
y_test = [y_test.iloc[i] for i in range(len(y_test))]
print(y_test, type(y_test))

rf_acc = accuracy_score(y_test, rf_y_predict)
dt_acc = accuracy_score(y_test, dt_y_predict)
lr_reg_acc = accuracy_score(y_test, lr_reg_y_predict)

# ---------- START ROC PLOT ------------------
# Convert predictions to probabilities
rf_y_proba = rf_classifier.predict_proba(X_test)
dt_y_proba = dt_classifier.predict_proba(X_test)
lr_reg_y_proba = lr_reg_classifier.predict_proba(X_test)

# Apply binarizer to compute OvR ROC Plot
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)

# We have 3 models
ylabs = ["HER2+", "HR+", "Triple Neg"]
roc_titles = ["Random Forest, AUC = 0.94", "Decision Tree, AUC = 0.8", "Logistic Regression, AUC = 0.96"]
y_proba_lst = [rf_y_proba, dt_y_proba, lr_reg_y_proba]

# Initialize figures
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle("ROC Plot One vs Rest (OvR)")

# For each model
for i in range(len(y_proba_lst)):
    # For each class
    axs[i].grid(color='gray', linestyle='--', linewidth=0.25)
    for j in range(len(ylabs)):
        # Compute Area Under the Curve
        curve_rf = roc_curve(y_onehot_test[:, j], y_proba_lst[i][:, j])
        auc_rf = auc(curve_rf[0], curve_rf[1])

        axs[i].plot(curve_rf[0], curve_rf[1], label=f'AUC ({ylabs[j]}) = %0.2f' % auc_rf, linewidth=2.5)
        axs[i].plot([0, 1], [0, 1], color='gray', linestyle='--')
        axs[i].set_xlabel('FPR')
        axs[i].set_ylabel('TPR')

    axs[i].set_title(roc_titles[i])
    axs[i].legend(loc='best', bbox_to_anchor=(0.4, 0.23))
plt.show()

# ---------- END ROC PLOT ------------------

# Calculate confusion matrices
rf_confusion_matrix = confusion_matrix(y_test, rf_y_predict)
dt_confusion_matrix = confusion_matrix(y_test, dt_y_predict)
lr_reg_confusion_matrix = confusion_matrix(y_test, lr_reg_y_predict)

# Define display labels
display_labels = ["HER2+", "HR+", "Triple Neg"]

# Create subplots
fig, axs = plt.subplots(figsize=(8,6))
# Plot confusion matrices
cm = ConfusionMatrixDisplay(confusion_matrix=rf_confusion_matrix, display_labels=display_labels)
cm.plot(ax=axs, cmap='Blues')

axs.grid(False)
axs.set_title(f'Random Forest, accuracy: {rf_acc}')

# Show plot
plt.show()

# Feature importance
importance = rf_classifier.feature_importances_
names_features = X_train.columns

# plot feature importance
forest_importances = pd.Series(importance, index=names_features)
sort_forest_importances = forest_importances.sort_values(ascending = False)
top_forest_importances = sort_forest_importances[:5]

print(top_forest_importances)

fig, ax = plt.subplots()
top_forest_importances.plot.bar(ax=ax)

ax.set_title("RF Feature Importance")
ax.set_ylabel("Mean Decrease in Impurity")
fig.tight_layout()
plt.show()
