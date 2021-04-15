from iii_utils import *
from iii_a_first_split_data import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import accuracy_score
from sklearn.model_selection import HalvingGridSearchCV


X = X.drop(['user_id', 'srch_destination_id'], axis=1)
holdout_X = holdout_X.drop(['user_id', 'srch_destination_id'], axis=1)

# Find best decision tree
param_grid = {
    'max_depth': [10, 20, 30, 40, 50],
}
clf = tree.DecisionTreeClassifier(random_state=6)
model = HalvingGridSearchCV(clf, param_grid, cv=5, factor=2).fit(X, y)
print(model.best_estimator_)

# fit best decision tree
clf = tree.DecisionTreeClassifier(random_state=6, max_depth=20)
clf.fit(X, y)

# predict
y_pred = clf.predict(holdout_X)

# evaluate
get_accuracy(holdout_y, y_pred, 'Best Decision Tree Classifier')

# Find best random forest classifier
param_grid = {
    'max_depth': [30, 40, 50],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=6)
model = HalvingGridSearchCV(rf, param_grid, cv=5, factor=2, resource='n_estimators', max_resources=30).fit(X, y)
print(model.best_estimator_)

# fit best rf classifier
rfc = RandomForestClassifier(max_depth=40, min_samples_split=5, n_estimators=24, random_state=6)
rfc.fit(X, y)

# predict
y_pred = rfc.predict(holdout_X)

# Get top features
df_features = pd.DataFrame(zip(X.columns, list(rfc.feature_importances_)),
                           columns=['Feature Name', 'Importance Value'])
top_features = df_features.sort_values(by='Importance Value',
                                          ascending=False).iloc[:25, :].loc[:, "Feature Name"].tolist()

# Evaluate
print(f'Accuracy Score for Random Forest: {accuracy_score(holdout_y, y_pred):.2%}')

get_pickled_model('clf.sav', clf)
get_pickled_model('rfc.sav', rfc)

pickle.dump(top_features, open('top_features.sav', 'wb'))