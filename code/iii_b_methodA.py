from iii_utils import *
from iii_a_first_split_data import *
from sklearn.linear_model import LogisticRegression

top_features = pickle.load(open('top_features.sav', 'rb'))
# only use balanced variables

balanced_vars = ['is_mobile',
    'is_package', 'is_booking',
    'Cin_day', 'Cin_day_of_week', 'Cin_week', 'Cin_year', 'cnt_bin',
    'posa_continent_combined_cats',
    'user_location_country_combined_cats',
    'user_location_region_combined_cats', 'channel_combined_cats',
    'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
    'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
    'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
    'no_days_to_cin_bin_combined_cats', 'site_name_combined_cats', 'hotel_market']

balanced_indices = [train.columns.get_loc(c) for c in balanced_vars]
print(balanced_indices)

top_tau_with_hotel_cluster = ['is_package', 'hotel_continent_combined_cats', 'hotel_country_combined_cats',
                              'hotel_market']


# Save original X datasets
X_ = X.copy()
holdout_X_ = holdout_X.copy()

# Logistic Regression with all balanced variables
X = X[balanced_vars]

# Fit
model_full = LogisticRegression(random_state=6, multi_class='multinomial', max_iter=2000)
model_full.fit(X, y)

# Predictions
holdout_X = holdout_X[balanced_vars]
y_pred = model_full.predict(holdout_X)

# Evaluate the model
get_accuracy(holdout_y, y_pred, 'Full Logistic Model')

# Model with top 10 features from Random Forest
X_ = X_[top_features]
model_rf_top_vars = LogisticRegression(random_state=6, multi_class='multinomial', max_iter=2000)
model_rf_top_vars.fit(X_, y)

# Predictions
holdout_X_ = holdout_X_[top_features]
y_pred = model_rf_top_vars.predict(holdout_X_)

# Evaluate the model
get_accuracy(holdout_y, y_pred, 'Logistic Model with Top Features from Random Forest')

# Model with top tau
model_top_tau = LogisticRegression(random_state=7, multi_class='multinomial', max_iter=2000)
X_top_tau = X[top_tau_with_hotel_cluster]
model_top_tau.fit(X_top_tau, y)

# Predictions
holdout_X_top_tau = holdout_X[top_tau_with_hotel_cluster]
y_pred_v = model_top_tau.predict(holdout_X_top_tau)

# Evaluate the model
get_accuracy(holdout_y, y_pred_v, 'Logistic Model with Top Tau Correlations with Hotel Cluster')

get_pickled_model('model_rf_top_vars.sav', model_rf_top_vars)
get_pickled_model('model_top_tau.sav', model_top_tau)