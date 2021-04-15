from ii_b_wrangle_data import *
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, precision_score
import seaborn as sns
from shared_functions import *
import pandas as pd
import pickle
import os


def save_fig(fname, verbose=True):
    path = os.path.join('figs', fname)
    plt.savefig(path, bbox_inches='tight')
    if verbose:
        print("Figure saved as '{}'".format(path))


test = pd.read_pickle('test.pkl')
print('test')
cmap = sns.diverging_palette(400, 200, 100, as_cmap=True)


clf = pickle.load(open('clf.sav', 'rb'))
rfc = pickle.load(open('rfc.sav', 'rb'))
rbf = pickle.load(open('rbf.sav', 'rb'))
top_features = pickle.load(open('top_features.sav', 'rb'))
model_rf_top_vars = pickle.load(open('model_rf_top_vars.sav', 'rb'))
model_top_tau = pickle.load(open('model_top_tau.sav', 'rb'))
top_tau_with_hotel_cluster = pickle.load(open('top_tau_with_hotel_cluster.sav', 'rb'))
balanced_vars = pickle.load(open('balanced_vars.sav', 'rb'))

'''
@inputs: name of model, model description, prediction probability of y, labels, alpha level, and the test y 
array
@outputs: dataframe
@purpose: concatenates results for prediction interval and coverage into a clear output
'''


def get_coverage_result(model_name, model_description, y_pred_prob, labels, alpha, test_y):
    pred_interval = category_pred_interval(y_pred_prob, labels, alpha, test_y)
    coverage_table = coverage(pred_interval)
    class_list = [key for key, value in coverage_table['avg_len'].items()]
    df_coverage = pd.DataFrame({
        'Model Name': [model_name] * len(class_list),
        'Model Description': [model_description] * len(class_list),
        'Class': class_list,
        'Probability': [alpha] * len(class_list),
        'Average Length': [np.round(value, decimals=4) for key, value in coverage_table['avg_len'].items()],
        'Miss': [np.round(value, decimals=4) for key, value in coverage_table['miss'].items()],
        'Miss Rate': [np.round(value, decimals=4) for key, value in coverage_table['miss_rate'].items()],
        'Coverage Rate': [np.round(value, decimals=4) for key, value in coverage_table['cov_rate'].items()]
    })
    return df_coverage


'''
@inputs: name of model, model description, arrays for train x, train y, test x, test y
@outputs: results data frame, coverage data frame for 50, coverage data frame for 80 prediction interval
@purpose: gets further results such as AUC for model
'''


def get_result(model, model_name, model_description, train_x, train_y, test_x, test_y):
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
    disp.ax_.set_title(f'{model_name} {model_description},out_of_sample_pred')
    save_fig(f'cv figs/{model_name} {model_description}, out_of_sample_pred')

    # Get coverage results for each class of the model
    df_coverage_50 = get_coverage_result(model_name, model_description, y_pred_prob,
                                         labels=labels, alpha=0.5, test_y=test_y)

    df_coverage_80 = get_coverage_result(model_name, model_description, y_pred_prob,
                                         labels=labels, alpha=0.8, test_y=test_y)

    # Get overall model results
    result = pd.DataFrame(
        {
            "Model Name": [model_name],
            "Model Description": [model_description],
            "Accuracy Score": [np.round(accuracy_score(test_y, y_pred), decimals=4)],
            "Precision Score": [np.round(precision_score(test_y, y_pred, average='macro'), decimals=4)],
            "AUC": [np.round(auc, decimals=4)]
        }
    )
    return result, df_coverage_50, df_coverage_80


df_results = pd.DataFrame()
results = pd.DataFrame()

prediction_interval_50 = pd.DataFrame()
prediction_interval_80 = pd.DataFrame()

X_train = train.drop(['hotel_cluster'], axis=1)
X_test = test.drop(['hotel_cluster'], axis=1).to_numpy()
columns = list(X_train.columns)
X_train = X_train.to_numpy()

X_train_svm = train.drop(['hotel_cluster'], axis=1)
y_train = train['hotel_cluster'].to_numpy()
index_test_start = X_train_svm.shape[0]

X_test_svm = test.drop(['hotel_cluster'], axis=1)
y_test = test['hotel_cluster'].to_numpy()

X_svm = pd.concat([X_train_svm, X_test_svm], axis=0)
columns = list(X_train_svm.columns)
X_svm_= X_svm[balanced_vars]
X_svm = pd.get_dummies(X_svm)
X_svm = X_svm.to_numpy()

X_train_svm = X_svm[:index_test_start,]
X_test_svm = X_svm[index_test_start:,]


# 1st Model: Decision Tree with depth = 30
model_result, model_50, model_80 = get_result(clf, "Decision Tree", "Depth = 30",
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

print(results.head())
# 2nd Model: Random Forest
model_result, model_50, model_80 = get_result(rfc, "Random Forest", "Depth = 40, Min Splits = 5 ",
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
model_result, model_50, model_80 = get_result(model_rf_top_vars, "Multinomial Logistic Regression",
                                              "Top Features Chosen by RandomForest",
                                              X_train[:, [columns.index(col) for col in top_features]],
                                              y_train,
                                              X_test[:, [columns.index(col) for col in top_features]],
                                              y_test)
results = results.append(model_result, ignore_index=True)
prediction_interval_50 = prediction_interval_50.append(model_50, ignore_index=True)
prediction_interval_80 = prediction_interval_80.append(model_80, ignore_index=True)

# 4th Model: Multinomial Logistic Regression with Top Features from Top Tau
model_result, model_50, model_80 = get_result(model_top_tau, "Multinomial Logistic Regression",
                                              "Top Features Chosen from Top Tau",
                                              X_train[:, [columns.index(col) for col in top_tau_with_hotel_cluster]],
                                              y_train,
                                              X_test[:, [columns.index(col) for col in top_tau_with_hotel_cluster]],
                                              y_test)
results = results.append(model_result, ignore_index=True)
prediction_interval_50 = prediction_interval_50.append(model_50, ignore_index=True)
prediction_interval_80 = prediction_interval_80.append(model_80, ignore_index=True)

# 5th Model: RBF SVM, drop user_id, srch_destination_id, then get_dummies on the rest of variables
model_result, model_50, model_80 = get_result(rbf, "SVM", "rbf",
                                              X_train_svm,
                                              y_train,
                                              X_test_svm,
                                              y_test)
results = results.append(model_result, ignore_index=True)
prediction_interval_50 = prediction_interval_50.append(model_50, ignore_index=True)
prediction_interval_80 = prediction_interval_80.append(model_80, ignore_index=True)

results.reset_index(drop=True)
results.to_pickle('svm_model_results_out_of_sample.pkl')
prediction_interval_50.reset_index(drop=True)
prediction_interval_80.reset_index(drop=True)

prediction_interval_80.to_pickle('svm_pred_int_80_out_of_sample.pkl')
prediction_interval_50.to_pickle('svm_pred_int_50_out_of_sample.pkl')

results_grouped = results.groupby([
    'Model Name',
    'Model Description'])[['Accuracy Score',
                           'Precision Score',
                           'AUC']].agg('mean').reset_index().to_csv('svm_model_results_out_of_sample_grouped.csv',
                                                                    index=False)

pred_int_50_grouped = prediction_interval_50.groupby([
    'Model Name',
    'Model Description',
    "Class"])[['Average Length', 'Miss', 'Miss Rate',
               'Coverage Rate']].agg('mean').reset_index().to_csv('svm_prediction_intervals_50_out_of_sample_grouped.csv',
                                                                  index=False)

pred_int_80_grouped = prediction_interval_80.groupby([
    'Model Name',
    'Model Description',
    "Class"])[['Average Length', 'Miss', 'Miss Rate',
               'Coverage Rate']].agg('mean').reset_index().to_csv('svm_prediction_intervals_80_out_of_sample_grouped.csv',
                                                                  index=False)

# Load saved dfs
# results = pd.read_pickle('model_results_out_of_sample.pkl')
# prediction_interval_50 = pd.read_pickle('pred_int_50_out_of_sample.pkl')
# prediction_interval_80 = pd.read_pickle('pred_int_80_out_of_sample.pkl')
