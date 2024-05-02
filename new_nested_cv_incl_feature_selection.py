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
rf_hp_grid = {'n_estimators': [1, 5, 10, 20, 50, 100],
              'max_depth': [5, 10, 15]}
dt_hp_grid = {'max_depth': [5, 10, 15],
              'max_features': [8, 16, 20, 25, 30, "sqrt", "log2"]}
# lr_hp_grid = {'C': [0.001, 0.01, 0.1, 0.7, 2, 3, 10, 100], 'penalty': ['l2', 'None'],
#                'solver': ['newton-cg', 'sag', 'saga', 'lbfgs']}
lr_hp_grid = {'C': [0.001, 0.01, 0.1, 0.7, 2, 3, 10, 100], 'penalty': ['l1', 'l2', 'None'],
               'solver': ['saga']}



# CV technique for outer and inner folds
outer_cv = StratifiedKFold(n_splits=4)
inner_cv = StratifiedKFold(n_splits=6)

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
lg_fold_performance_lst = []

dt_fold_hp_lst = []  # For df and rf we will append the best hyperparameters to these lists
rf_fold_hp_lst = []
lg_fold_hp_lst = []

for i, (train_index, test_index) in enumerate(outer_cv.split(X_train, y_train)):
    print(f"We are currently in Outer fold {i + 1}")
    X_train_fold = X_train.iloc[train_index, :]  # train_index is a list of indices, but we can pass lists of indices in np
    y_train_fold = y_train[train_index]
    X_validate_fold = X_train.iloc[test_index, :]
    y_validate_fold = y_train[test_index]

    # ------START: Step 3) Feature selection (unfinished) ------
        # Note that when we have this step figured out, we need to change the data that we train on in the next step


    ### INITIAL FEATURE SELECTION ###
    # Combining X_train and y_train again in one dataframe
    y_train_df = pd.DataFrame(y_train_fold)
    train_data = X_train_fold
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

    significant = []
    for i in range(0, len(df_chi['p-value'])):
        if df_chi.iloc[i,2] <= 0.05:    # when p-value is smaller than or equal to critical value
            significant.append(True)
        else:
            significant.append(False)

    df_chi['significant'] = significant

    # Select the significant features
    significant_features = df_chi[df_chi['significant']]
    features = pd.DataFrame(significant_features['Feature'])

    features_list = list(features['Feature'])
    
    # Making correlation matrix
    #selected = X_train.iloc[:, features_list]
    #correlation_matrix = selected.corr()

    # Make for each feature an empty set
    clusters_list = [set() for i in range(len(features_list) + 1)]
    j = 0
    for i in range(0, len(features_list)):
        feature_i_data = X_train.iloc[:, i]
        feature_neighbour_data = X_train.iloc[:, i + 1]
        current_cluster = clusters_list[j]
        if abs(feature_i_data.corr(feature_neighbour_data)) > 0.9:
            # print("Feature",i,"and feature", i+1, "are correlated")
            current_cluster.add(features_list[i])
            if i != (len(features_list)-1):
                current_cluster.add(features_list[i + 1])
        else:
            # print("Feature",i,"and feature", i+1, "are not correlated")
            j += 1  # we need to go to a new cluster

    clusters_list = [cluster for cluster in clusters_list if cluster != set()]
    print(len(clusters_list))

    # Make a total list of all correlated features and a list of all non-correlated features
    correlated_features = []
    for cluster in clusters_list:
        for feature in cluster:
            correlated_features.append(feature)
    non_correlated_features = []
    for feature in features_list:
        if feature not in correlated_features:
            non_correlated_features.append(feature)

    # from each cluster we randomly pick one feature
    indep_features_list = []
    for cluster in clusters_list:
        feature_random = random.choice(list(cluster))
        indep_features_list.append(feature_random)
    indep_features_list.extend(non_correlated_features)
    print('This fold has', len(indep_features_list) ,'independent features')
    # select the X_train_fold data for only independent features
    r_X_train_fold = X_train_fold.iloc[:, indep_features_list]
    r_X_validate_fold = X_validate_fold.iloc[:, indep_features_list]

   
    # ------END: Step 3) Feature selection (unfinished) ------



    # Step 4) Apply 5CV Grid search to each X_train_fold

    # Define classifiers
    rf_classifier = RandomForestClassifier(random_state=42)
    dt_classifier = DecisionTreeClassifier(random_state=42)
    lg_classifier = LogisticRegression(random_state=42, multi_class='multinomial')

    # Define grid search object for all classifiers
    # USE ACCURACY AS A PLACEHOLDER MEASURE!!! (to be removed)
    rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_hp_grid, cv=inner_cv, scoring='accuracy',
                                  verbose=3)
    dt_grid_search = GridSearchCV(estimator=dt_classifier, param_grid=dt_hp_grid, cv=inner_cv, scoring='accuracy',
                                  verbose=3)
    lr_grid_search = GridSearchCV(estimator=lg_classifier, param_grid=lr_hp_grid, cv=inner_cv, scoring='accuracy',
                                    verbose=3)

    # Run grid search for all classifiers on the training data of this current fold
    rf_grid_search.fit(r_X_train_fold, y_train_fold)
    dt_grid_search.fit(r_X_train_fold, y_train_fold)
    lr_grid_search.fit(r_X_train_fold, y_train_fold)

    # Extract the most important parameter and the corresponding score
    rf_best_hp = rf_grid_search.best_params_
    rf_best_score = rf_grid_search.best_score_


    dt_best_hp = dt_grid_search.best_params_
    dt_best_score = dt_grid_search.best_score_

    lr_best_hp = lr_grid_search.best_params_ 
    lr_best_score = lr_grid_search.best_score_



    # store the best hyperparameters in the dictionaries
    rf_fold_hp_lst.append([rf_best_hp, rf_best_score])
    dt_fold_hp_lst.append([dt_best_hp, dt_best_score])
    lg_fold_hp_lst.append([lr_best_hp, lr_best_score])

    print("Random forest best hp:", rf_best_hp, "Score:", rf_best_score)
    print("Decision tree best hp:", dt_best_hp, "Score:", dt_best_score)
    print("Logistic regression best hp:", lr_best_hp, "Score:", lr_best_score)

    # Define new models with the optimal hyperparameters
    best_rf_classifier = rf_grid_search.best_estimator_
    best_dt_classifier = dt_grid_search.best_estimator_
    best_lr_classifier = lr_grid_search.best_estimator_

    # Step 5) Now that we have the best models for this fold, we can test the performance on X_train_fold
    rf_y_predict_fold = best_rf_classifier.predict(r_X_validate_fold)
    dt_y_predict_fold = best_dt_classifier.predict(r_X_validate_fold)
    lr_y_predict_fold = best_lr_classifier.predict(r_X_validate_fold)

    # Retrieve the accuracy score
    rf_accuracy = accuracy_score(rf_y_predict_fold, y_validate_fold)
    dt_accuracy = accuracy_score(dt_y_predict_fold, y_validate_fold)
    lr_accuracy = accuracy_score(lr_y_predict_fold, y_validate_fold)

    # Append performance to list
    rf_fold_performance_lst.append(rf_accuracy)
    dt_fold_performance_lst.append(dt_accuracy)
    lg_fold_performance_lst.append(lr_accuracy)



    MSE_values = []
    C_values = [0.001, 0.01, 0.1, 0.7, 2, 3, 10, 100]

    for C in C_values:
        model = LogisticRegression(C=C, penalty = lr_best_hp['penalty'], solver = lr_best_hp['solver'], multi_class='multinomial',
                                   random_state=42, max_iter=1000)
        model.fit(r_X_train_fold, y_train_fold)
        y_ = model.predict(r_X_validate_fold)
        mse = np.mean((y_ - y_validate_fold) ** 2)
        MSE_values.append(mse)

    plt.plot(C_values, MSE_values, marker='o')
    plt.xlabel('C (Penalization Strength)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE vs C for Logistic Regression')
    plt.xscale('log')
    plt.grid(True)
plt.show()


print("Performance of random forest per fold", rf_fold_performance_lst, "Average performance:", mean(rf_fold_performance_lst))
print("Performance of decision tree per fold", dt_fold_performance_lst, "Average performance:", mean(dt_fold_performance_lst))
print("Performance of logistic regression per fold", lg_fold_performance_lst, "Average performance:", mean(lg_fold_performance_lst))


# print("Best hyperparameters for random forest per fold", rf_fold_hp_lst)
# print("Best hyperparameters for decision tree per fold", dt_fold_hp_lst)
# print("Best hyperparameters for logistic regression per fold", lg_fold_hp_lst)


   


