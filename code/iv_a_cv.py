from ii_b_wrangle_data import *
from ii_c_exploratory_analysis import save_fig
from iii_c_methodB import clf, rfc, top_features
from iii_b_methodA import lm_fs
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, precision_score
import seaborn as sns
from shared_functions import *


# Join first 25000 rows and randomize the dataframe
data = data.append(test)
data = shuffle(data, random_state=42)
data = data.reset_index(drop=True)
cmap = sns.diverging_palette(400, 200, 100, as_cmap=True)

print(top_features)


def get_result(model, model_name, model_description, i, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    y_pred_prob = model.predict_proba(test_x)
    auc = roc_auc_score(test_y, y_pred_prob, multi_class='ovo')
    plt.figure()
    labels = [41, 48, 64, 65, 91]
    disp = plot_confusion_matrix(model, test_x, test_y, display_labels=labels,
                                 cmap=cmap)
    disp.ax_.set_title(f'{model_name} {model_description}, Fold {str(i)}')
    save_fig(f'{model_name} {model_description}, Fold {str(i)}')
    pred_interval_50 = category_pred_interval(y_pred_prob, labels, 0.5, test_y, ".")
    pred_interval_50_results = coverage(pred_interval_50)
    print(pred_interval_50_results)
    pred_interval_80 = category_pred_interval(y_pred_prob, labels, 0.8, test_y, ".")
    pred_interval_80_results = coverage(pred_interval_80)

    result = pd.DataFrame(
        {
            "Model Name": [model_name],
            "Description": [model_description],
            "Fold": [i],
            "Accuracy Score": [np.round(accuracy_score(test_y, y_pred), decimals=4)],
            "Precision Score": [np.round(precision_score(test_y, y_pred, average='macro'), decimals=4)],
            "AUC": [np.round(auc, decimals=4)],
            "Average Length 50% Prediction Interval": [np.round(pred_interval_50_results["avg_len"],decimals=4)],
            "Average Length 80% Prediction Interval": [np.round(pred_interval_80_results["avg_len"],decimals=4)],
            "Coverage Rate 50% Prediction Interval": [np.round(pred_interval_50_results["cov_rate"],decimals=4)],
            "Coverage Rate 80% Prediction Interval": [np.round(pred_interval_80_results["cov_rate"],decimals=4)],
        }
    )
    return result


# Do cross-validation
df_results = pd.DataFrame()
n_folds = 5
kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
results = pd.DataFrame()
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
    # X_train = pd.DataFrame(data=X_train, columns=columns)
    # X_test = pd.DataFrame(data=X_test, columns=columns)
    # y_train = pd.DataFrame(data=y_train, columns=['hotel_cluster'])
    # y_test = pd.DataFrame(data=y_test, columns=['hotel_cluster'])

    # 1st Model: Decision Tree with depth = 30
    results = results.append(get_result(clf, "Decision Tree", "Depth = 30", fold,
                                        np.delete(X_train,
                                                  np.s_[columns.index('user_id'),
                                                        columns.index('srch_destination_id')], axis=1),
                                        y_train,
                                        np.delete(X_test,
                                                  np.s_[columns.index('user_id'),
                                                        columns.index('srch_destination_id')], axis=1),
                                        y_test), ignore_index=True)

    # clf.fit(np.delete(X_train, np.s_[columns.index('user_id'), columns.index('srch_destination_id')], axis=1),
    #         y_train,)
    # clf_disp = plot_roc_curve(clf, np.delete(X_test,
    #                                          np.s_[columns.index('user_id'),
    #                                                columns.index('srch_destination_id')], axis=1), y_test,
    #                           response_method='predict_proba')

    # 2nd Model: Random Forest
    results = results.append(get_result(rfc, "Random Forest", "Depth = 40, Min Splits = 5 ", fold,
                                        np.delete(X_train,
                                                  np.s_[columns.index('user_id'),
                                                        columns.index('srch_destination_id')], axis=1),
                                        y_train,
                                        np.delete(X_test,
                                                  np.s_[columns.index('user_id'),
                                                        columns.index('srch_destination_id')], axis=1),
                                        y_test), ignore_index=True)
    # rfc.fit(np.delete(X_train, np.s_[columns.index('user_id'), columns.index('srch_destination_id')], axis=1),
    #         y_train)
    # rfc_disp = plot_roc_curve(rfc, np.delete(X_test,
    #                                          np.s_[columns.index('user_id'),
    #                                                columns.index('srch_destination_id')], axis=1), y_test, ax=ax)

    # 3rd Model: Logistic Regression with Top Features Used by Random Forest
    results = results.append(get_result(lm_fs, "Logistic Regression", "Top Features Chosen by RandomForest", fold,
                                        X_train[:, [columns.index(col) for col in top_features]],
                                        y_train,
                                        X_test[:, [columns.index(col) for col in top_features]],
                                        y_test))
    # lm_fs.fit(X_train[:, [columns.index(col) for col in top_features]], y_train)
    # lm_fs_disp = plot_roc_curve(lm_fs, X_test[:, [columns.index(col) for col in top_features]], y_test, ax=ax)

    # Plot ROC for 3 models
    fold = fold + 1
    # plt.figure()
    # clf_disp.plot(ax=ax, alpha=0.8)
    # save_fig("ROC")

results.reset_index(drop=True)
results.to_csv("CV RESULTS.csv", index=False)


