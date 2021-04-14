from ii_b_wrangle_data import *
from ii_c_exploratory_analysis import save_fig
from iii_b_methodA import lm_fs
from iii_c_methodB import clf, rfc, top_features
# from iii_d_methodC import linear
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, precision_score
import seaborn as sns
from shared_functions import *
import pandas as pd

# Join first rows and randomize the dataframe
data = shuffle(train, random_state=42)
data = data.reset_index(drop=True)
cmap = sns.diverging_palette(400, 200, 100, as_cmap=True)


def get_coverage_result(model_name, model_description, fold_i, y_pred_prob, labels, alpha, test_y):
    pred_interval = category_pred_interval(y_pred_prob, labels, alpha, test_y)
    coverage_table = coverage(pred_interval)
    class_list = [key for key, value in coverage_table['avg_len'].items()]
    df_coverage = pd.DataFrame({
        'Model Name': [model_name] * len(class_list),
        'Model Description': [model_description] * len(class_list),
        'Fold': [fold_i] * len(class_list),
        'Class': class_list,
        'Probability': [alpha] * len(class_list),
        'Average Length': [np.round(value, decimals=4) for key, value in coverage_table['avg_len'].items()],
        'Miss': [np.round(value, decimals=4) for key, value in coverage_table['miss'].items()],
        'Miss Rate': [np.round(value, decimals=4) for key, value in coverage_table['miss_rate'].items()],
        'Coverage Rate': [np.round(value, decimals=4) for key, value in coverage_table['cov_rate'].items()]
    })
    return df_coverage


def get_result(model, model_name, model_description, fold, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)

    # Get predictions
    y_pred = model.predict(test_x)
    y_pred_prob = model.predict_proba(test_x)

    # Get AUC
    auc = roc_auc_score(test_y, y_pred_prob, multi_class='ovo')

    # Plot Confusion Matrix
    plt.figure()
    labels = [41, 48, 64, 65, 91]
    disp = plot_confusion_matrix(model, test_x, test_y, display_labels=labels,
                                 cmap=cmap)
    disp.ax_.set_title(f'{model_name} {model_description}, Fold {str(fold)}')
    save_fig(f'cv figs/{model_name} {model_description}, Fold {str(fold)}')

    # Get coverage results for each class of the model
    df_coverage_50 = get_coverage_result(model_name, model_description, fold, y_pred_prob,
                                         labels=labels, alpha=0.5, test_y=test_y)

    df_coverage_80 = get_coverage_result(model_name, model_description, fold, y_pred_prob,
                                         labels=labels, alpha=0.8, test_y=test_y)

    # Get overall model results
    result = pd.DataFrame(
        {
            "Model Name": [model_name],
            "Model Description": [model_description],
            "Fold": [fold],
            "Accuracy Score": [np.round(accuracy_score(test_y, y_pred), decimals=4)],
            "Precision Score": [np.round(precision_score(test_y, y_pred, average='macro'), decimals=4)],
            "AUC": [np.round(auc, decimals=4)]
        }
    )
    return result, df_coverage_50, df_coverage_80


# Do cross-validation
df_results = pd.DataFrame()
n_folds = 5
kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
results = pd.DataFrame()

prediction_interval_50 = pd.DataFrame()
prediction_interval_80 = pd.DataFrame()

fold = 0

for train_index, test_index in kf.split(data):
    X = data.drop(['hotel_cluster'], axis=1)
    columns = list(X.columns)
    X = X.to_numpy()
    y = data['hotel_cluster'].to_numpy()
    print(f'shape of test index {test_index.shape}')
    print(f'shape of train index {train_index.shape}')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 1st Model: Decision Tree with depth = 30
    model_result, model_50, model_80 = get_result(clf, "Decision Tree", "Depth = 30", fold,
                                                  np.delete(X_train, np.s_[columns.index('user_id'),
                                                                           columns.index('srch_destination_id')],
                                                            axis=1),
                                                  y_train,
                                                  np.delete(X_test,
                                                            np.s_[columns.index('user_id'),
                                                                  columns.index('srch_destination_id')], axis=1),
                                                  y_test)

    results = results.append(model_result, ignore_index=True)
    prediction_interval_50 = prediction_interval_50.append(model_50, ignore_index=True)
    prediction_interval_80 = prediction_interval_80.append(model_80, ignore_index=True)

    # 2nd Model: Random Forest
    model_result, model_50, model_80 = get_result(rfc, "Random Forest", "Depth = 40, Min Splits = 5 ", fold,
                                                  np.delete(X_train,
                                                            np.s_[columns.index('user_id'),
                                                                  columns.index('srch_destination_id')], axis=1),
                                                  y_train,
                                                  np.delete(X_test,
                                                            np.s_[columns.index('user_id'),
                                                                  columns.index('srch_destination_id')], axis=1),
                                                  y_test)

    results = results.append(model_result, ignore_index=True)
    prediction_interval_50 = prediction_interval_50.append(model_50, ignore_index=True)
    prediction_interval_80 = prediction_interval_80.append(model_80, ignore_index=True)

    # 3rd Model: Logistic Regression with Top Features Used by Random Forest
    model_result, model_50, model_80 = get_result(lm_fs, "Logistic Regression", "Top Features Chosen by RandomForest",
                                                  fold,
                                                  X_train[:, [columns.index(col) for col in top_features]],
                                                  y_train,
                                                  X_test[:, [columns.index(col) for col in top_features]],
                                                  y_test)
    results = results.append(model_result, ignore_index=True)
    prediction_interval_50 = prediction_interval_50.append(model_50, ignore_index=True)
    prediction_interval_80 = prediction_interval_80.append(model_80, ignore_index=True)

    # 4th Model: Linear SVM, drop user_id, srch_destination_id, then get_dummies on the rest of variables
    # model_result, model_50, model_80 = get_result(linear, "SVM", "Linear Kernel",
    #                                               fold,
    #                                               pd.get_dummies(np.delete(X_train,
    #                                                                        np.s_[columns.index('user_id'),
    #                                                                              columns.index('srch_destination_id')],
    #                                                                        axis=1)),
    #                                               y_train,
    #                                               pd.get_dummies(np.delete(X_test,
    #                                                                        np.s_[columns.index('user_id'),
    #                                                                              columns.index('srch_destination_id')],
    #                                                                        axis=1)),
    #                                               y_test)
    # results = results.append(model_result, ignore_index=True)
    # prediction_interval_50 = prediction_interval_50.append(model_50, ignore_index=True)
    # prediction_interval_80 = prediction_interval_80.append(model_80, ignore_index=True)
    #
    fold = fold + 1

results.reset_index(drop=True)
results.to_pickle('model_results.pkl')
prediction_interval_50.reset_index(drop=True)
prediction_interval_80.reset_index(drop=True)

prediction_interval_80.to_pickle('pred_int_80.pkl')
prediction_interval_50.to_pickle('pred_int_50.pkl')

# Load saved dfs
# results = pd.read_pickle('model_results.pkl')
# prediction_interval_50 = pd.read_pickle('pred_int_50.pkl')
# prediction_interval_80 = pd.read_pickle('pred_int_80.pkl')

# Average results over number folds
results_grouped = results.groupby([
    'Model Name', 'Model Description'])[['Accuracy Score',
                                         'Precision Score',
                                         'AUC']].agg('mean').reset_index().to_csv('Model Results Grouped.csv',
                                                                                  index=False)

pred_int_50_grouped = prediction_interval_50.groupby([
    'Model Name',
    'Model Description',
    "Class"])[['Average Length', 'Miss', 'Miss Rate',
               'Coverage Rate']].agg('mean').reset_index().to_csv('Prediction Intervals 50% Grouped.csv', index=False)

pred_int_80_grouped = prediction_interval_80.groupby([
    'Model Name',
    'Model Description',
    "Class"])[['Average Length', 'Miss', 'Miss Rate',
               'Coverage Rate']].agg('mean').reset_index().to_csv('Prediction Intervals 80% Grouped.csv', index=False)
