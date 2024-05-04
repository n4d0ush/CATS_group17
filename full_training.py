import pandas as pd
import numpy as np
import random
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from statistics import mean

from sklearn.model_selection import GridSearchCV, \
    RandomizedSearchCV  # Hyperparameter tuning - GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, \
    KFold  # Cross-validation - StratifiedKFold, RepeatedStratifiedKFold

from sklearn.ensemble import RandomForestClassifier  # Classifier - Random Forest
from sklearn.linear_model import LogisticRegression  # Classifier - Logistic Regression
from sklearn.tree import DecisionTreeClassifier  # Classifier - Decision Tree

from sklearn.metrics import roc_auc_score  # Evaluation metric - AUC
from sklearn.metrics import f1_score  # Evaluation metric - F1 score
from sklearn.metrics import accuracy_score  # Evaluation metric - Accuracy

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

# Hyperparameter grid
rf_hp_grid = {'n_estimators': [10, 20]}

dt_hp_grid = {
    'max_depth': [None, 10]}

lr_hp_grid_regularized = {
    'C': np.logspace(-4, 4, 10),
    'penalty': ['l2'],
    'solver': ['newton-cg', 'sag', 'saga', 'lbfgs'],
    'max_iter': [2000, 5000]
}

lr_hp_grid_non_regularized = {
    'C': [1],  # Doesn't matter since penalty=None
    'penalty': [None],
    'solver': ['newton-cg', 'sag', 'saga', 'lbfgs'],
    'max_iter': [2000, 5000]
}

# CV technique for outer and inner folds
outer_cv = StratifiedKFold(n_splits=4)
inner_cv = StratifiedKFold(n_splits=3)

# -----------
# Plan:
# 1) We already have a train/test split
# 2) Cross validation: we split the training data into 4-folds: train + validation
# 3) For each fold, we apply feature  selection to reduce the dataset
# 4) For each fold, we apply 6-CV for hyperparameter tuning
# 5) For each fold, we test the optimal models

# Step 1) Is already done
# Step 2) We split the data into 4 folds (outer folds)
y_train.reset_index(drop=True, inplace=True)  # We need to reset the indices of y_train so we can apply CV split
dt_fold_performance_lst = []  # For df and rf we will append their performances to these lists and then take the mean of these
rf_fold_performance_lst = []
lr_fold_performance_lst_regul = []
lr_fold_performance_lst_no_regul = []

dt_fold_hp_lst = []  # For df and rf we will append the best hyperparameters to these lists
rf_fold_hp_lst = []
lr_fold_hp_lst_regul = []
lr_fold_hp_lst_no_regul = []

# Create variables that store the final scores, hps and training features to use
final_score = -1
final_hps = None
final_feats = None
final_model = ""
final_top10_feats = None
dt_final_hp = None
lr_unreg_final_hp = None
lr_reg_final_hp = None

# Crossvalidation
for i, (train_index, test_index) in enumerate(outer_cv.split(X_train, y_train)):
    print(f"We are currently in Outer fold {i + 1}")
    X_train_fold = X_train.iloc[train_index,
                   :]  # train_index is a list of indices, but we can pass lists of indices in np
    y_train_fold = y_train[train_index]
    X_validate_fold = X_train.iloc[test_index, :]
    y_validate_fold = y_train[test_index]

    # Step 3) Feature selection

    # Combining X_train and y_train again in one dataframe
    y_train_df = pd.DataFrame(y_train_fold)
    train_data = X_train_fold.copy()
    train_data['Subgroup'] = list(y_train_df.iloc[:, 0])

    # Perform Chi-Squared test for each feature (column)
    chi2_results = []
    target_variable = 'Subgroup'

    # For each feature, find correlation with target variable
    for feature in train_data.columns:
        if feature != target_variable:  # Check if the feature is categorical
            contingency_table = pd.crosstab(train_data[feature], train_data[target_variable])
            results = chi2_contingency(contingency_table)
            chi2_results.append({'Feature': feature, 'Chi-Squared': results[0], 'p-value': results[1]})

    df_chi = pd.DataFrame(chi2_results)  # Save the results into a dataframe

    # Filter out the features with insignificant target correlations
    significant = []
    for j in range(0, len(df_chi['p-value'])):
        if df_chi.iloc[j, 2] <= 0.05:  # when p-value is smaller than or equal to critical value
            significant.append(True)
        else:
            significant.append(False)

    # Append whether feature is significant (True/False) to chi dataframe
    df_chi['significant'] = significant

    # Select the significant features
    significant_features = df_chi[df_chi['significant']]
    features = pd.DataFrame(significant_features['Feature'])
    features_list = list(features['Feature'])

    # Make for each feature an empty set
    clusters_list = [set() for i in range(len(features_list) + 1)]
    k = 0

    # Go through each significant feature
    for j in range(len(features_list) - 1):
        # Retrieve feature information
        feature_j_data = X_train.iloc[:, features_list[j]]
        feature_neighbour_data = X_train.iloc[:, features_list[j + 1]]
        current_cluster = clusters_list[k]

        # If cluster is empty, add current feature to it
        if len(current_cluster) == 0:
            current_cluster.add(features_list[j])

        # If neighboring significant feature has high correlation, add to current cluster. Else, create new cluster
        if abs(feature_j_data.corr(feature_neighbour_data)) > 0.85:
            # print("Feature",features_list[j],"and feature",features_list[j+1], "are correlated")
            current_cluster.add(features_list[j + 1])
        else:
            # print("Feature",features_list[j],"and feature", features_list[j+1], "are not correlated")
            k += 1  # we need to go to a new cluster

            # If the final feature does not correlate, we still need to create its own cluster
            if features_list[j + 1] == features_list[-1]:
                current_cluster = clusters_list[k]
                current_cluster.add(features_list[j + 1])

    # Remove all the empty clusters from clusters_list
    clusters_list = [cluster for cluster in clusters_list if cluster != set()]

    # from each cluster we randomly pick one feature
    indep_features_list = []
    for cluster in clusters_list:
        feature_random = random.choice(list(cluster))
        indep_features_list.append(feature_random)

    print('This fold has', len(indep_features_list), 'independent features')
    # select the X_train_fold data for only independent features
    r_X_train_fold = X_train_fold.iloc[:, indep_features_list]
    r_X_validate_fold = X_validate_fold.iloc[:, indep_features_list]

    # Step 4) Apply 5CV Grid search to each X_train_fold

    # Define classifiers
    rf_classifier = RandomForestClassifier(random_state=42)
    dt_classifier = DecisionTreeClassifier(random_state=42)
    lr_classifier = LogisticRegression(random_state=42, multi_class='multinomial')

    # Define grid search object for all classifiers
    rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_hp_grid, cv=inner_cv, scoring='accuracy',
                                  verbose=3)
    dt_grid_search = GridSearchCV(estimator=dt_classifier, param_grid=dt_hp_grid, cv=inner_cv, scoring='accuracy',
                                  verbose=3)

    lr_grid_search_regul = GridSearchCV(estimator=lr_classifier, param_grid=lr_hp_grid_regularized, cv=inner_cv,
                                        scoring='accuracy',
                                        verbose=3)
    lr_grid_search_no_regul = GridSearchCV(estimator=lr_classifier, param_grid=lr_hp_grid_non_regularized, cv=inner_cv,
                                           scoring='accuracy',
                                           verbose=3)

    # Run grid search for all classifiers on the training data of this current fold
    rf_grid_search.fit(r_X_train_fold, y_train_fold)
    dt_grid_search.fit(r_X_train_fold, y_train_fold)
    lr_grid_search_regul.fit(r_X_train_fold, y_train_fold)
    lr_grid_search_no_regul.fit(r_X_train_fold, y_train_fold)

    # Extract the most important parameter and the corresponding score
    rf_best_hp = rf_grid_search.best_params_
    rf_best_score = rf_grid_search.best_score_

    dt_best_hp = dt_grid_search.best_params_
    dt_best_score = dt_grid_search.best_score_

    lr_best_hp_regul = lr_grid_search_regul.best_params_
    lr_best_score_regul = lr_grid_search_regul.best_score_

    lr_best_hp_no_regul = lr_grid_search_no_regul.best_params_
    lr_best_score_no_regul = lr_grid_search_no_regul.best_score_

    # store the best hyperparameters in the dictionaries
    rf_fold_hp_lst.append([rf_best_hp, rf_best_score])
    dt_fold_hp_lst.append([dt_best_hp, dt_best_score])
    lr_fold_hp_lst_regul.append([lr_best_hp_regul, lr_best_score_regul])
    lr_fold_hp_lst_no_regul.append([lr_best_hp_no_regul, lr_best_score_no_regul])

    # Define new models with the optimal hyperparameters
    best_rf_classifier = rf_grid_search.best_estimator_
    best_dt_classifier = dt_grid_search.best_estimator_
    best_lr_classifier_regul = lr_grid_search_regul.best_estimator_
    best_lr_classifier_no_regul = lr_grid_search_no_regul.best_estimator_

    # Step 5) Now that we have the best models for this fold, we can test the performance on X_train_fold
    rf_y_predict_fold = best_rf_classifier.predict(r_X_validate_fold)
    dt_y_predict_fold = best_dt_classifier.predict(r_X_validate_fold)
    lr_y_predict_fold_regul = best_lr_classifier_regul.predict(r_X_validate_fold)
    lr_y_predict_fold_no_regul = best_lr_classifier_no_regul.predict(r_X_validate_fold)

    # Retrieve the accuracy score
    rf_accuracy = accuracy_score(rf_y_predict_fold, y_validate_fold)
    dt_accuracy = accuracy_score(dt_y_predict_fold, y_validate_fold)
    lr_accuracy_regul = accuracy_score(lr_y_predict_fold_regul, y_validate_fold)
    lr_accuracy_no_regul = accuracy_score(lr_y_predict_fold_no_regul, y_validate_fold)

    print("Random forest best hp:", rf_best_hp, "Score on validation fold:", rf_accuracy)
    print("Decision tree best hp:", dt_best_hp, "Score on validation fold:", dt_accuracy)
    print("Logistic regression best hp, with regularization:", lr_best_hp_regul,
          "Score on validation fold, with regularization:", lr_accuracy_regul)
    print("Logistic regression best hp, no regularization:", lr_best_hp_no_regul,
          "Score on validation fold, no regularization:", lr_accuracy_no_regul)

    # Append performance to list
    rf_fold_performance_lst.append(rf_accuracy)
    dt_fold_performance_lst.append(dt_accuracy)
    lr_fold_performance_lst_regul.append(lr_accuracy_regul)
    lr_fold_performance_lst_no_regul.append(lr_accuracy_no_regul)

    # Feature selection: find top 10
    rf_importance = best_rf_classifier.feature_importances_
    rf_names_features = r_X_train_fold.columns

    dt_importance = best_dt_classifier.feature_importances_
    dt_names_features = r_X_train_fold.columns

    lr_importance_regul = best_lr_classifier_regul.coef_
    lr_importance_avg_regul = np.mean(np.abs(lr_importance_regul), axis=0)
    # lr_names_features = r_X_train_fold.columns

    lr_importance_no_regul = best_lr_classifier_no_regul.coef_
    lr_importance_avg_no_regul = np.mean(np.abs(lr_importance_no_regul), axis=0)

    # Plot feature importance
    forest_importances = pd.Series(rf_importance, index=rf_names_features)
    sort_forest_importances = forest_importances.sort_values(ascending=False)
    top_forest_importances = sort_forest_importances[:10]

    tree_importances = pd.Series(dt_importance, index=dt_names_features)
    sort_tree_importances = tree_importances.sort_values(ascending=False)
    top_tree_importances = sort_tree_importances[:10]

    logistic_importances_regul = pd.Series(lr_importance_avg_regul)
    sort_logistic_importances_regul = logistic_importances_regul.sort_values(ascending=False)
    top_logistic_importances_regul = sort_logistic_importances_regul[:10]

    logistic_importances_no_regul = pd.Series(lr_importance_avg_no_regul)
    sort_logistic_importances_no_regul = logistic_importances_no_regul.sort_values(ascending=False)
    top_logistic_importances_no_regul = sort_logistic_importances_no_regul[:10]

    # Update the final hps, feats, scores if necessary
    if rf_accuracy > final_score:
        final_score = rf_accuracy
        final_hps = rf_best_hp
        final_feats = indep_features_list
        final_model = "Random forest"
        final_top10_feats = top_forest_importances
        dt_final_hp = dt_best_hp
        lr_unreg_final_hp = lr_best_hp_no_regul
        lr_reg_final_hp = lr_best_hp_regul

print("Performance of random forest per fold", rf_fold_performance_lst, "Average performance:",
      mean(rf_fold_performance_lst))
print("Performance of decision tree per fold", dt_fold_performance_lst, "Average performance:",
      mean(dt_fold_performance_lst))
print("Performance of logistic regression with regularization per fold", lr_fold_performance_lst_regul,
      "Average performance:",
      mean(lr_fold_performance_lst_regul))
print("Performance of logistic regression without regularization per fold", lr_fold_performance_lst_no_regul,
      "Average performance:",
      mean(lr_fold_performance_lst_no_regul))

print("\nSummary:")
print("Model:", final_model)
print("Score:", final_score)
print("Hyperparameters (RF):", final_hps)
print("Hyperparameters (DT):", dt_final_hp)
print("Hyperparameters (unreg LR):", lr_unreg_final_hp)
print("Hyperparameters (reg LR):", lr_reg_final_hp)
print("Features (RF):", final_feats)
print("Top 10 features (RF):", final_top10_feats)