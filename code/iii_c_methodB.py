import pandas as pd
from ii_c_exploratory_analysis import save_fig
from shared_functions import *
from iii_a_first_split_data import *
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from matplotlib import pyplot as plt
#
# balanced_vars = ['is_mobile',
#     'is_package', 'is_booking',
#     'Cin_day', 'Cin_day_of_week', 'Cin_week', 'Cin_year', 'cnt_bin',
#     'posa_continent_combined_cats',
#     'user_location_country_combined_cats',
#     'user_location_region_combined_cats', 'channel_combined_cats',
#     'srch_adults_cnt_combined_cats', 'srch_children_cnt_combined_cats',
#     'srch_rm_cnt_combined_cats', 'srch_destination_type_id_combined_cats', 'hotel_continent_combined_cats',
#     'hotel_country_combined_cats', 'stay_dur_bin_combined_cats',
#     'no_days_to_cin_bin_combined_cats', 'site_name_combined_cats']

# Fit a decision tree
X = X.drop(['user_id', 'srch_destination_id'], axis=1)
holdout_X = holdout_X.drop(['user_id', 'srch_destination_id'], axis=1)

clf = tree.DecisionTreeClassifier(random_state=6, max_depth=30)
clf.fit(X, y)
y_pred = clf.predict(holdout_X)
# category_pred_interval(clf.predict_proba(holdout_X), [41, 48, 64, 65, 91], 0.5, holdout_y,
#                        "Decision Tree - 50% Prediction Interval")
# category_pred_interval(clf.predict_proba(holdout_X), [41, 48, 64, 65, 91], 0.8, holdout_y,
#                        "Decision Tree - 80% Prediction Interval")
# plt.figure(figsize=(40, 40))
# tree.plot_tree(clf, feature_names=X.columns)
# save_fig("Decision Tree with max_depth=10")
# print(f'Accuracy Score for the Decision Tree: {accuracy_score(holdout_y, y_pred):.2%}')
# confusion_matrix_dt = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(confusion_matrix_dt)


# Fit Random Forest
# param_grid = {
#     'max_depth': [30, 40, 50],
#     'min_samples_split': [2, 5, 10]
# }
# rf = RandomForestClassifier(random_state=6)
# model = HalvingGridSearchCV(rf, param_grid, cv=5, factor=2, resource='n_estimators', max_resources=30).fit(X, y)
# print(model.best_estimator_)
# yields RandomForestClassifier(max_depth=40, min_samples_split=5, n_estimators=24, random_state=6)

rfc = RandomForestClassifier(max_depth=40, min_samples_split=5, n_estimators=24, random_state=6)
rfc.fit(X, y)
y_pred = rfc.predict(holdout_X)
df_features = pd.DataFrame(zip(X.columns, list(rfc.feature_importances_)),
                           columns=['Feature Name', 'Importance Value'])
top_features = df_features.sort_values(by='Importance Value',
                                          ascending=False).iloc[:25, :].loc[:, "Feature Name"].tolist()


# category_pred_interval(model.predict_proba(holdout_X), [41, 48, 64, 65, 91], 0.5, holdout_y,
#                        "Random Forest 50% - Prediction Interval")
# category_pred_interval(model.predict_proba(holdout_X), [41, 48, 64, 65, 91], 0.5, holdout_y,
#                        "Random Forest 80% - Prediction Interval")
# print(f'Accuracy Score for Random Forest: {accuracy_score(holdout_y, y_pred):.2%}')
# confusion_matrix_rf = pd.crosstab(holdout_y, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print(confusion_matrix_rf)
