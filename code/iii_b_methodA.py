from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from iii_a_first_split_data import *
from iii_c_methodB import top_features
from shared_functions import category_pred_interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# only use balanced variables

binned_vars = ['posa_continent_combined_cats',
               'user_location_country_combined_cats',
               'user_location_region_combined_cats', 'channel_combined_cats',
               'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
               'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
               'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
               'no_days_to_cin_bin_combined_cats', 'site_name_combined_cats']
#
# balanced_vars = [
#        'user_location_city', 'user_id', 'is_mobile',
#        'is_package', 'srch_destination_id',
#        'is_booking', 'hotel_market',
#        'Cin_day', 'Cin_day_of_week', 'Cin_week',
#        'Cin_month', 'Cin_year', 'cnt_bin',
#        'posa_continent_combined_cats',
#        'user_location_country_combined_cats',
#        'user_location_region_combined_cats', 'channel_combined_cats',
#        'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
#        'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
#        'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
#        'no_days_to_cin_bin_combined_cats', 'site_name_combined_cats']


# print(f'Before dropping columns of X: {X.columns}')
# X = X.drop(balanced_vars, axis=1)
# print(f'After dropping columns of X: {X.columns}')
# print(X.describe())
# model = LogisticRegression(random_state = 6, multi_class='multinomial')
# model.fit(X,y)
# holdout_X = holdout_X.drop(balanced_vars, axis=1)
# y_pred = model.predict(holdout_X)
# confusion_matrix = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(confusion_matrix)

# 41 and 48 clusters are predicted poorly, but 64, 65, 91 are predicted better

# # SKLearn Score : 45.12%
# score = model.score(holdout_X, holdout_y)
# print(score)
#
# X = X[balanced_vars]
# X = X.drop('user_id', axis=1)
#
# model = LogisticRegression(random_state = 6, multi_class='multinomial')
# model.fit(X, y)
#
# holdout_X = holdout_X[balanced_vars]
# holdout_X = holdout_X.drop('user_id', axis=1)
# y_pred = model.predict(holdout_X)
# y_pred_probs = model.predict_proba(holdout_X)
#
# print(y_pred)
# confusion_matrix = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(confusion_matrix)
#
# score = model.score(holdout_X, holdout_y)
# print(score)


# balanced_vars = [
#     'user_location_city', 'is_mobile',
#     'is_package',
#     'is_booking', 'hotel_market',
#     'Cin_day', 'Cin_day_of_week', 'Cin_week',
#     'Cin_month', 'Cin_year', 'cnt_bin',
#     'posa_continent_combined_cats',
#     'user_location_country_combined_cats',
#     'user_location_region_combined_cats', 'channel_combined_cats',
#     'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
#     'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
#     'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
#     'no_days_to_cin_bin_combined_cats', 'site_name_combined_cats']


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

for col in balanced_vars:
    for x in X[col]:
        if type(x) != str and x < 0:
            print(col)

# Save original X datasets
X_ = X.copy()
holdout_X_ = holdout_X.copy()

# Logistic Regression with all balanced variables
X = X[balanced_vars]

# Fit
model = LogisticRegression(random_state=6, multi_class='multinomial', max_iter=2000)
model.fit(X, y)

# Predictions
holdout_X = holdout_X[balanced_vars]
y_pred = model.predict(holdout_X)
category_pred_interval(model.predict_proba(holdout_X), [41, 48, 64, 65, 91], 0.5, holdout_y)
category_pred_interval(model.predict_proba(holdout_X), [41, 48, 64, 65, 91], 0.8, holdout_y)


# Evaluate the model
confusion_matrix_lm_full = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
accuracy_lm_full = accuracy_score(holdout_y, y_pred)
print(f'Full Logistic Model Accuracy: {accuracy_lm_full:.2%}')


# Model with top 10 features from Logistic Regression
lm_fs = LogisticRegression(random_state=6, multi_class='multinomial', max_iter=2000)
X_ = X_[top_features]
lm_fs.fit(X_, y)
# Predictions
holdout_X_ = holdout_X_[top_features]
y_pred = lm_fs.predict(holdout_X_)
category_pred_interval(lm_fs.predict_proba(holdout_X_), [41, 48, 64, 65, 91], 0.5, holdout_y)
category_pred_interval(lm_fs.predict_proba(holdout_X_), [41, 48, 64, 65, 91], 0.8, holdout_y)


# Evaluate the model
confusion_matrix_lm_fs = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
accuracy_lm_fs = accuracy_score(holdout_y, y_pred)
print(f'Logistic Model with Top Features from Random Forest Accuracy: {accuracy_lm_fs:.2%}')  # 51.89%

# # Logistic Regression with Model Built using Chi-Squared Features
# def select_features(X_train, y_train, X_test, criteria, k):
#     if criteria == "Chi-Squared":
#         feature_selection = SelectKBest(score_func=chi2, k=k)
#     else:
#         feature_selection = SelectKBest(score_func=mutual_info_classif, k=k)
#     feature_selection.fit(X_train, y_train)
#     X_train_fs = feature_selection.transform(X_train)
#     X_test_fs = feature_selection.transform(X_test)
#     return X_train_fs, X_test_fs
#
#
# accuracy_results_for_fs = pd.DataFrame({
#     "Model": ["Logistic Regression"],
#     "Number of Features": [len(X.columns)],
#     "Feature Selection": ["None"],
#     "Accuracy Score": [np.round(accuracy_lm_full, decimals=4)]
# })

t = []
s = []
# for n in range(1, len(balanced_vars)):
#     for selection in ["Chi-Squared", "Mutual Information"]:
#         X_fs, holdout_X_fs = select_features(X, y, holdout_X, selection, n)
#
#         # Fit a logistic model with selected features
#         model.fit(X_fs, y)
#
#         # Get predictions for the logistic model with selected features
#         y_pred_fs = model.predict(holdout_X_fs)
#
#         # Evaluate the logistic model with selected features
#         confusion_matrix_lm_fs = pd.crosstab(holdout_y, y_pred_fs, rownames=['Actual'], colnames=['Predicted'])
#         accuracy_lm_fs = accuracy_score(holdout_y, y_pred_fs)
#         print(f'Accuracy of Logistic Model with {n}-Selected Features using {selection}: {accuracy_lm_fs:.2%}')
#         row = pd.DataFrame({
#             "Model": ["Logistic Regression"],
#             "Number of Features": [n],
#             "Feature Selection": [selection],
#             "Accuracy Score": [np.round(accuracy_lm_fs, decimals=4)]
#         })
#         accuracy_results_for_fs = accuracy_results_for_fs.append(row, ignore_index=True)
#
#         if selection == "Chi-Squared":
#             t.append(accuracy_lm_fs)
#         else:
#             s.append(accuracy_lm_fs)


# v = range(1, len(balanced_vars))
# plt.plot(v, t)
# plt.savefig('figs/chi-square-selection')
#
# plt.plot(v,s)
# plt.savefig('figs/mutual-inf-sel')

# accuracy_results_for_fs = accuracy_results_for_fs.reset_index(drop=True)
# accuracy_results_for_fs.to_csv("Accuracy Score for Logistic Models.csv", index=False)
