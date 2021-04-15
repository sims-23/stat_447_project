from sklearn.metrics import accuracy_score
import pickle

'''
@inputs: holdout_y array, pred_y array, model description string
@outputs: None
@purpose: prints accuracy score between actual values and predicted values
'''
def get_accuracy(holdout_y, pred_y, model_descrip):
    accuracy = accuracy_score(holdout_y, pred_y)
    print(model_descrip + f' Accuracy: {accuracy:.2%}')

'''
@inputs: file name, model that is fitted
@outputs: None
@purpose: creates a picked model object to use later to filename
'''
def get_pickled_model(filename, model):
    pickle.dump(model, open(filename, 'wb'))