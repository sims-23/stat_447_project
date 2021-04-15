from sklearn.metrics import accuracy_score

'''
@inputs: holdout_y array, pred_y array, model description string
@outputs: None
@purpose: prints accuracy score between actual values and predicted values
'''
def get_accuracy(holdout_y, pred_y, model_descrip):
    accuracy = accuracy_score(holdout_y, pred_y)
    print(model_descrip + f' Accuracy: {accuracy:.2%}')
