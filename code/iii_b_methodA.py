from sklearn.linear_model import LogisticRegression
from iii_a_first_split_data import *
import pandas as pd

#only use balanced variables

binned_vars = ['posa_continent_combined_cats',
       'user_location_country_combined_cats',
       'user_location_region_combined_cats', 'channel_combined_cats',
       'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
       'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
       'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
       'no_days_to_cin_bin_combined_cats', 'site_name_combined_cats']

balanced_vars = [
       'user_location_city', 'user_id', 'is_mobile',
       'is_package', 'srch_destination_id',
       'is_booking', 'hotel_market',
       'Cin_day', 'Cin_day_of_week', 'Cin_week',
       'Cin_month', 'Cin_year', 'cnt_bin',
       'posa_continent_combined_cats',
       'user_location_country_combined_cats',
       'user_location_region_combined_cats', 'channel_combined_cats',
       'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
       'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
       'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
       'no_days_to_cin_bin_combined_cats', 'site_name_combined_cats']

X = X.drop(balanced_vars, axis=1)

model = LogisticRegression(random_state = 6, multi_class='multinomial')
model.fit(X,y)
holdout_X = holdout_X.drop(balanced_vars, axis=1)
y_pred = model.predict(holdout_X)
confusion_matrix = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# X = X[balanced_vars]
#
# model = LogisticRegression(random_state = 4, multi_class='multinomial')
# model.fit(X,y)
#
# holdout_X = holdout_X[balanced_vars]
# y_pred = model.predict(holdout_X)
# y_pred_probs = model.predict_proba(holdout_X)
#
# print(y_pred)
# confusion_matrix = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(confusion_matrix)


