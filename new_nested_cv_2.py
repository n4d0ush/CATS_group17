import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

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


    # # Select the significant features
    significant_features = df_chi[df_chi['significant']]
    # print(len(significant_features))
    # --> 10 

   
    features = pd.DataFrame(significant_features['Feature'])
    # features.to_csv('output_feature_selection.txt', index= False)
    


    #select the feature in X_train_fold given by the 'Feature' column in features
    # r_X_train_fold = X_train_fold.iloc[:, features['Feature']]



     # Extract the indices of significant features
    selected_feature_indices = significant_features['Feature'].index

    # Select the significant features from X_train_fold
    r_X_train_fold = X_train_fold.iloc[:, selected_feature_indices]



   
    # ------END: Step 3) Feature selection (unfinished) ------



    # Step 4) Apply 5CV Grid search to each X_train_fold

    # Define classifiers
    rf_classifier = RandomForestClassifier(random_state=42)
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Define grid search object for all classifiers
    # USE ACCURACY AS A PLACEHOLDER MEASURE!!! (to be removed)
    rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_hp_grid, cv=inner_cv, scoring='accuracy',
                                  verbose=3)
    dt_grid_search = GridSearchCV(estimator=dt_classifier, param_grid=dt_hp_grid, cv=inner_cv, scoring='accuracy',
                                  verbose=3)

    # Run grid search for all classifiers on the training data of this current fold
    rf_grid_search.fit(r_X_train_fold, y_train_fold)
    dt_grid_search.fit(r_X_train_fold, y_train_fold)

    # Extract the most important parameter and the corresponding score
    rf_best_hp = rf_grid_search.best_params_
    rf_best_score = rf_grid_search.best_score_

    dt_best_hp = dt_grid_search.best_params_
    dt_best_score = dt_grid_search.best_score_

    print("Random forest best hp:", rf_best_hp, "Score:", rf_best_score)
    print("Decision tree best hp:", dt_best_hp, "Score:", dt_best_score)

    # Define new models with the optimal hyperparameters
    best_rf_classifier = rf_grid_search.best_estimator_
    best_dt_classifier = dt_grid_search.best_estimator_

    # Step 5) Now that we have the best models for this fold, we can test the performance on X_train_fold
    rf_y_predict_fold = best_rf_classifier.predict(X_validate_fold)
    dt_y_predict_fold = best_dt_classifier.predict(X_validate_fold)

    # Retrieve the accuracy score
    rf_accuracy = accuracy_score(rf_y_predict_fold, y_validate_fold)
    dt_accuracy = accuracy_score(dt_y_predict_fold, y_validate_fold)

    # Append performance to list
    rf_fold_performance_lst.append(rf_accuracy)
    dt_fold_performance_lst.append(dt_accuracy)

print("Performance of random forest per fold", rf_fold_performance_lst, "Average performance:", mean(rf_fold_performance_lst))
print("Performance of decision tree per fold", dt_fold_performance_lst, "Average performance:", mean(dt_fold_performance_lst))
