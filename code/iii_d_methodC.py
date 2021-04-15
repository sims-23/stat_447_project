# SVM
from sklearn import svm
from iii_a_first_split_data import *
import pandas as pd
from sklearn.model_selection import GridSearchCV
from iii_utils import get_pickled_model
import pickle

# Toggle value to avoid extensive running
GET_BEST_PARAM = False

balanced_vars = ['is_mobile',
    'is_package', 'is_booking',
    'Cin_day', 'Cin_day_of_week', 'Cin_week', 'Cin_year', 'cnt_bin',
    'posa_continent_combined_cats',
    'user_location_country_combined_cats',
    'user_location_region_combined_cats', 'channel_combined_cats',
    'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
    'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
    'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
    'no_days_to_cin_bin_combined_cats', 'site_name_combined_cats']


X = X[balanced_vars]
holdout_X = holdout_X[balanced_vars]

# must get dummies for SVM to work
X = pd.get_dummies(X)
holdout_X = pd.get_dummies(holdout_X)

def svc_param_selection(X, y, kernel, param_grid):
    svc = svm.SVC(decision_function_shape='ovo', probability=True, kernel=kernel)
    grid_search = GridSearchCV(svc, param_grid)
    grid_search.fit(X, y)
    print(kernel + " best params are")
    print(grid_search.best_params_)

if GET_BEST_PARAM:
    param_grid_linear = [
        {'C': [0.001, 0.01, 0.1, 1, 10]},
    ]

    svc_param_selection(X, y, 'linear', param_grid_linear)

    param_grid_rbf = [
        {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
    ]

    svc_param_selection(X, y, 'rbf', param_grid_rbf)


# one vs one SVM for multiclassification linear with best parameters
linear = svm.SVC(kernel='linear', C=0.01, decision_function_shape='ovo', probability=True)
linear.fit(X, y)
accuracy_lin = linear.score(holdout_X, holdout_y)
print('Accuracy Linear Kernel:', accuracy_lin)

# one vs one SVM for multiclassification rbf with best parameters
rbf = svm.SVC(kernel='rbf', gamma=0.1, C=10, decision_function_shape='ovo', probability=True).fit(X, y)
accuracy_rbf = rbf.score(holdout_X, holdout_y)
print('Accuracy RBF Kernel:', accuracy_rbf)

# get best model as pickled object
get_pickled_model('rbf.sav', rbf)
pickle.dump(balanced_vars, open('balanced_vars.sav', 'wb'))