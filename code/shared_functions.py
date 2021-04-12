import pandas as pd
import numpy as np

# ' @description
# ' Prediction intervals for a categorical response
# ' @param ProbMatrix of dimension nxJ, J = # categories,
# ' each row is a probability mass function
# ' @param labels vector of length J, with short names for categories
# '
# ' @details
# ' A more general function can be written so the levels of prediction intervals
# ' can be other than 0.50 and 0.80.
# '
# ' @return list with two string vectors of length n:
# ' pred50 has 50% prediction intervals
# ' pred80 has 80% prediction intervals
#
"""
CategoryPredInterval = function(ProbMatrix,labels)
{ ncases=nrow(ProbMatrix)
pred50=rep(NA,ncases); pred80=rep(NA,ncases)
for(i in 1:ncases)
{ p=ProbMatrix[i,]
ip=order(p)
pOrdered=p[ip] # increasing order
labelsOrdered=labels[rev(ip)] # decreasing order
G=rev(cumsum(c(0,pOrdered))) # cumulative sum from smallest
k1=min(which(G<=0.5))-1 # 1-level1= 1-0.5=0.5
k2=min(which(G<=0.2))-1 # 1-level2= 1-0.8=0.2
pred1=labelsOrdered[1:k1]; pred2=labelsOrdered[1:k2]
pred50[i]=paste(pred1,collapse="")
pred80[i]=paste(pred2,collapse="")
}
list(pred50=pred50, pred80=pred80)
}
"""


# Given the row from a dataframe, get the order of values if the row was sorted in the increasing order
def get_order(row):
    list_row = row.values.tolist()
    temp = sorted(list_row)
    return [temp.index(i) for i in list_row]


# Construct Prediction Intervals
def category_pred_interval(prob_matrix, labels, prob_value=0.5):
    n_cases = prob_matrix.shape[0]
    pred_list: list = [1] * n_cases
    for i in range(0, n_cases):
        p = prob_matrix.iloc[i, :]
        print(p.values)
        ip = get_order(p)
        print(ip)
        p_ordered = p[ip].values.tolist()
        print(p_ordered)
        labels_ordered = [labels[i] for i in ip[::-1]]
        print(labels_ordered)
        g = list(np.cumsum([0] + p_ordered))[::-1]
        print(f'G: {g}')
        k = np.min(np.where(np.array(g) <= prob_value)) - 1
        print(f'{np.where(np.array(g) <= prob_value)}')
        pred_labels = list(labels_ordered[:k+1])
        print(f'pred labels: {pred_labels}')
        pred_list[i] = '.'.join(pred_labels)
        if i == 1:
            break
    return pred_list


probMatrix = {'1': [4.082367e-10, 1.822106e-15, 1.000000e+00],
              '2': [1.315919e-03, 9.986605e-01, 2.359995e-05],
              '3': [9.279982e-01, 7.195826e-02, 4.356354e-05],
              '4': [4.277739e-08, 5.927599e-10, 1.000000e+00],
              '5': [9.980396e-01, 1.888962e-03, 7.143053e-05],
              '6': [2.143369e-08, 1.411895e-10, 1.000000e+00]
              }
correctProbMatrix = pd.DataFrame(np.transpose(pd.DataFrame(probMatrix).to_numpy()))
correctProbMatrix = correctProbMatrix.set_index(pd.Series([12, 43, 42, 23, 54, 28]))
print(f'Correct Prob Matrix:\n{correctProbMatrix}')
# print(np.array([1.00000001895, 0.99868409995, 0.9986605, 0.0]))
# print(np.where(np.array([1.00000001895, 0.99868409995, 0.9986605, 0.0]) <= 0.5))
# print(np.mean(np.where(np.array([1.00000001895, 0.99868409995, 0.9986605, 0.0]) <= 0.5)))
# print(np.min(np.where(np.array([2, 4, 56, 12]) > 10)))
print(category_pred_interval(correctProbMatrix, ["B", "D", "P"]))
#