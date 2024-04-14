import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler                                # Standardization

from scipy.stats import f_oneway                                                # Feature selection - ANOVA
from sklearn.linear_model import Lasso, ElasticNet                              # Feature selection - Lasso and ElasticNet


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV            # Hyperparameter tuning - GridSearchCV, RandomizedSearchCV 
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold    # Cross-validation - StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score                             # Metric - cross_val_score outer loop

from sklearn.ensemble import RandomForestClassifier                             # Classifier - Random Forest
from sklearn.linear_model import LogisticRegression                             # Classifier - Logistic Regression
from sklearn.tree import DecisionTreeClassifier                                 # Classifier - Decision Tree


from sklearn.metrics import roc_auc_score                                       # Evaluation metric - AUC
from sklearn.metrics import f1_score                                            # Evaluation metric - F1 score
from sklearn.metrics import accuracy_score                                      # Evaluation metric - Accuracy




# Load data

data = pd.read_table('Train_call.txt')
data = data.T
data_target= pd.read_table('Train_clinical.txt')


# Extract predictor and target data
target = data_target.loc[:, "Subgroup"]
predictors = data.iloc[4:,:]


# Split data into training and testing sets
#x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2,random_state=42) 

X_train_trial, X_test_trial, y_train_trial, y_test_trial = train_test_split(predictors, target, test_size=0.2, random_state=42)



#shape of the data

print("The dimension of x_train is {}".format(X_train_trial.shape))
print("The dimension of x_test is {}".format(X_test_trial.shape))

#Scale features
scaler = StandardScaler()
X_train_0 = scaler.fit_transform(X_train_trial)
X_test_0 = scaler.transform(X_test_trial)
y_train_0 = scaler.fit_transform(y_train_trial)
y_test_0 = scaler.transform(y_test_trial)






# PARAMETER GRID Dictionary

model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10, 20, 50, 100],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='saga', multi_class='auto'),
        'params': {
            'C': [1, 5, 10, 20, 50, 100]
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }
    }
}

# CV tecnique
outer_cv = StratifiedKFold(n_splits=4, suffle = False, random_state=2)
inner_cv = StratifiedKFold(n_splits=6, suffle = False, random_state=2)

# Nested CV with parameter tunning: GridSearchCV and RandomizedSearchCV

# GridSearchCV for hyperparameter tuning  (Computational expensive when many permutation and combination for different parameters)
best_scores_GScv_per_model = {}
dataframes = {} 
nested_scores_GScv = {}



for model_name, mp in model_params.items():
    clf_GScv = GridSearchCV(mp['model'], mp['params'], cv=inner_cv, return_train_score=False) #clf = classifier

    # Results of the inner loop
    clf_GScv.cv_results_
    results = pd.DataFrame(clf_GScv.cv_results_) 
    dataframes[model_name] = results # save the results for all different splits in the inner loop
  
    best_scores = {'Best score': [clf_GScv.best_score_], 'Best params': clf_GScv.best_params_} 
    best_scores_df = pd.DataFrame(best_scores)
    best_scores_GScv_per_model[model_name]= best_scores_df # save the best parameters and score in the best_scores_GScv_per_model inner loop

    # Results of the outer loop
    nested_score_GScv = cross_val_score(clf_GScv, X=X_train_0, y=y_train_0, cv=outer_cv, scoring= 'roc_auc')
    mean_nested_score_GScv = np.mean(nested_score_GScv)
    nested_scores_GScv[model_name] = mean_nested_score_GScv # save the mean score of the nested loop in the nested_scores_GScv outer loop




   








# RandomizedSearchCV for hyperparameter tuning (Less computationally expensive than GridSearchCV. Give x random combinations of parameters, chosen by us.)
scores_RScv = []
for model_name, mp in model_params.items():
    clf_RScv = RandomizedSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, n_iter=2) #clf = classifier
    clf_RScv.fit(X_train_0, y_train_trial)
    
    #clf_RScv.cv_results_
    #df_all_results = pd.DataFrame(clf_RScv.cv_results_) # Dataframe of the results for all different splits

    scores_RScv.append({
        'model': model_name,
        'best_score': clf_RScv.best_score_,
        'best_params': clf_RScv.best_params_
    })

df_best_results_RScv = pd.DataFrame(scores_RScv, columns=['model', 'best_score', 'best_params'])




# Model object
rand_for_model = RandomForestClassifier()
log_reg_model = LogisticRegression()
dec_tree_model = DecisionTreeClassifier()




