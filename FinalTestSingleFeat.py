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
feats = [2184]
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

rf_classifier.fit(X_train, y_train)

# Test performance on test set
rf_y_predict = rf_classifier.predict(X_test)

# Compute accuracy
y_test = [y_test.iloc[i] for i in range(len(y_test))]

rf_acc = accuracy_score(y_test, rf_y_predict)

# ---------- START ROC PLOT ------------------
# Convert predictions to probabilities
rf_y_proba = rf_classifier.predict_proba(X_test)

# Apply binarizer to compute OvR ROC Plot
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)

# We have 3 models
ylabs = ["HER2+", "HR+", "Triple Neg"]
y_proba_lst = rf_y_proba
print(y_proba_lst)

# Initialize figures
fig, axs = plt.subplots(figsize=(5, 5))

# For each class
axs.grid(color='gray', linestyle='--', linewidth=0.25)
for j in range(len(ylabs)):
    # Compute Area Under the Curve
    curve_rf = roc_curve(y_onehot_test[:, j], y_proba_lst[:, j])
    auc_rf = auc(curve_rf[0], curve_rf[1])

    axs.plot(curve_rf[0], curve_rf[1], label=f'AUC ({ylabs[j]}) = %0.2f' % auc_rf, linewidth=2.5)
    axs.plot([0, 1], [0, 1], color='gray', linestyle='--')
    axs.set_xlabel("FPR")
    axs.set_ylabel("TPR")

axs.set_title("Random Forest (Best Feature), AUC=0.84")
axs.legend(loc='best', bbox_to_anchor=(0.42, 0.23))
plt.show()

# ---------- END ROC PLOT ------------------

# Calculate confusion matrices
rf_confusion_matrix = confusion_matrix(y_test, rf_y_predict)

# Define display labels
display_labels = ["HER2+", "HR+", "Triple Neg"]

# Create subplots
fig, axs = plt.subplots(figsize=(8,6))
# Plot confusion matrices
cm = ConfusionMatrixDisplay(confusion_matrix=rf_confusion_matrix, display_labels=display_labels)
cm.plot(ax=axs, cmap='Blues')

axs.grid(False)
axs.set_title(f'Random Forest (Best Feature), accuracy: {rf_acc}')

# Show plot
plt.show()

# Feature importance
importance = rf_classifier.feature_importances_
names_features = X_train.columns

# plot feature importance
forest_importances = pd.Series(importance, index=names_features)
sort_forest_importances = forest_importances.sort_values(ascending = False)
top_forest_importances = sort_forest_importances[:10]

print(top_forest_importances)

fig, ax = plt.subplots()
top_forest_importances.plot.bar(ax=ax)

ax.set_title("RF Feature Importance")
ax.set_ylabel("Mean Decrease in Impurity")
fig.tight_layout()