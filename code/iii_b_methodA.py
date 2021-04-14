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


# Model with top 10 features from Random Forest
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

