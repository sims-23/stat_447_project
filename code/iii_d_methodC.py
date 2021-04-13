# SVM
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from iii_a_first_split_data import *
from shared_functions import *
import pandas as pd
#
# balanced_vars = ['is_mobile',
#     'is_package', 'is_booking', 'Cin_day_of_week', 'Cin_week', 'cnt_bin',
#     'posa_continent_combined_cats',
#     'user_location_region_combined_cats', 'channel_combined_cats',
#     'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
#     'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
#     'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
#     'no_days_to_cin_bin_combined_cats']


X = X.drop(['user_id', 'srch_destination_id'], axis=1)
# X = X[balanced_vars]
holdout_X = holdout_X.drop(['user_id', 'srch_destination_id'], axis=1)
# holdout_X = holdout_X[balanced_vars]


# ohe = OneHotEncoder()
# ohe.fit_transform(X)
# print(X)
X = pd.get_dummies(X)
holdout_X = pd.get_dummies(holdout_X)
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
linear.fit(X, y)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X, y)
# poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X, y)
# sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X, y)

# retrieve the accuracy and print it for all 4 kernel functions
# accuracy_lin = linear.score(holdout_X, holdout_y)
# accuracy_poly = poly.score(holdout_X, holdout_y)
accuracy_rbf = rbf.score(holdout_X, holdout_y)
# accuracy_sig = sig.score(holdout_X, holdout_y)


# y_pred = linear.predict(holdout_X)
# print('Accuracy Linear Kernel:', accuracy_lin)
y_pred = rbf.predict(holdout_X)
print('Accuracy Polynomial Kernel:', accuracy_rbf)
confusion_matrix_linear_kernel_full = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix_linear_kernel_full)

category_pred_interval(linear.predict_proba(holdout_X), [41, 48, 64, 65, 91], 0.5, holdout_y)
category_pred_interval(linear.predict_proba(holdout_X), [41, 48, 64, 65, 91], 0.8, holdout_y)

# print('Accuracy Polynomial Kernel:', accuracy_poly)
# print('Accuracy Radial Basis Kernel:', accuracy_rbf)
# print('Accuracy Sigmoid Kernel:', accuracy_sig)
# 0.7382501807664498 with almost all features 
