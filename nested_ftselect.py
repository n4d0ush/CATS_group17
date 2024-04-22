import pandas as pd
import numpy as np

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error # Evaluation metric - Accuracy

from sklearn.linear_model import ElasticNet  # Regularization - ElasticNet

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


# #  Convert from series to dataframe
# y_train = y_train.to_frame()
# y_test = y_test.to_frame()

y_train = y_train.astype(int)  # For some reason the numbers are read as strings, so convert to integers
y_test = y_test.astype(int)



# Hyperparameter grid
rf_hp_grid = {'n_estimators': [1, 5, 10, 20, 50, 100],
              'max_depth': [5, 10, 15]}
dt_hp_grid = {'max_depth': [5, 10, 15],
              'max_features': [8, 16, 20, 25, 30, "sqrt", "log2"]}

# CV technique for outer and inner folds
outer_cv = StratifiedKFold(n_splits=4)
inner_cv = StratifiedKFold(n_splits=6)



# Hyperparmeter grid feature 
    # ElasticNet: regression model that combines L1 and L2 regularization 

en_hp_grid = {'alpha': [0.1, 0.5, 1, 2, 5, 10], # Constant that multiplies the penalty term. Determines the relative weight of L1 and L2 regularization.
                'l1_ratio': [0, 0.1, 0.5, 0.7, 0.9, 1]} # l1_ratio of 1 is Lasso (L1 penalty), 0 is Ridge (L2 penalty)

# Define ElasticNet model
en_model = ElasticNet(random_state=42)

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
    

    # Normzalize the data --> When penality is applied, it is important that the data is normalized
        #sklearn.preprocessing.StandardScaler

    # Define grid search object for ElasticNet
    cv_en = KFold(n_splits=5, shuffle=True, random_state=42)
    en_grid_search = GridSearchCV(estimator=en_model, param_grid=en_hp_grid, cv=cv_en, scoring='neg_mean_squared_error',
                                  verbose=3)
    # Run grid search for ElasticNet on the training data of this current fold
    en_grid_search.fit(X_train_fold, y_train_fold)

    # Extract the most important parameter and the corresponding score
    en_best_hp = en_grid_search.best_params_
    en_best_score = en_grid_search.best_score_
    en_best_estimator = en_grid_search.best_estimator_

    print("ElasticNet best hp:", en_best_hp, "Score:", en_best_score, "Best estimator:", en_best_estimator)

    # Define new model with the optimal hyperparameters
    en_opt_model = ElasticNet(alpha=en_best_hp['alpha'], l1_ratio=en_best_hp['l1_ratio'], random_state=42)

    # Fit the optimal model on the training data
    en_opt_model.fit(X_train_fold, y_train_fold)
    
    # Extract the coefficients of the optimal model. Saved in en_coefficients dictionary
    feat_fold = []
    for coef, feature in zip(en_opt_model.coef_, X_train_fold.columns):
        if coef != 0:
            feat_fold.append(feature)
    
    en_feat_df = pd.DataFrame(feat_fold)
    en_feat_df.to_csv(f"Feat_fold{i+1}.csv")

    #select the columns in X_train_fold that are in feat_fold
    X_train_fold_reduced = X_train_fold[feat_fold]
    


    # ------END: Step 3) Feature selection (unfinished) ------

#     # Step 4) Apply 6CV Grid search to each X_train_fold

#     # Define classifiers
#     rf_classifier = RandomForestClassifier(random_state=42)
#     dt_classifier = DecisionTreeClassifier(random_state=42)

#     # Define grid search object for all classifiers
#     # USE ACCURACY AS A PLACEHOLDER MEASURE!!! (to be removed)
#     rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_hp_grid, cv=inner_cv, scoring='accuracy',
#                                   verbose=3)
#     dt_grid_search = GridSearchCV(estimator=dt_classifier, param_grid=dt_hp_grid, cv=inner_cv, scoring='accuracy',
#                                   verbose=3)

#     # Run grid search for all classifiers on the training data of this current fold
#     rf_grid_search.fit(X_train_fold, y_train_fold)
#     dt_grid_search.fit(X_train_fold, y_train_fold)

#     # Extract the most important parameter and the corresponding score
#     rf_best_hp = rf_grid_search.best_params_
#     rf_best_score = rf_grid_search.best_score_

#     dt_best_hp = dt_grid_search.best_params_
#     dt_best_score = dt_grid_search.best_score_

#     print("Random forest best hp:", rf_best_hp, "Score:", rf_best_score)
#     print("Decision tree best hp:", dt_best_hp, "Score:", dt_best_score)

#     # Define new models with the optimal hyperparameters
#     best_rf_classifier = rf_grid_search.best_estimator_
#     best_dt_classifier = dt_grid_search.best_estimator_

#     # Step 5) Now that we have the best models for this fold, we can test the performance on X_train_fold
#     rf_y_predict_fold = best_rf_classifier.predict(X_validate_fold)
#     dt_y_predict_fold = best_dt_classifier.predict(X_validate_fold)

#     # Retrieve the accuracy score
#     rf_accuracy = accuracy_score(rf_y_predict_fold, y_validate_fold)
#     dt_accuracy = accuracy_score(dt_y_predict_fold, y_validate_fold)

#     # Append performance to list
#     rf_fold_performance_lst.append(rf_accuracy)
#     dt_fold_performance_lst.append(dt_accuracy)

# print("Performance of random forest per fold", rf_fold_performance_lst, "Average performance:", mean(rf_fold_performance_lst))
# print("Performance of decision tree per fold", dt_fold_performance_lst, "Average performance:", mean(dt_fold_performance_lst))
