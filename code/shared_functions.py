import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

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

coverage=function(Table)
{ nclass=nrow(Table); nsubset=ncol(Table); rowFreq=rowSums(Table)
labels=rownames(Table); subsetLabels=colnames(Table)
cover=rep(0,nclass); avgLen=rep(0,nclass)
for(irow in 1:nclass)
{ for(icol in 1:nsubset)
{ intervalSize = nchar(subsetLabels[icol])
isCovered = grepl(labels[irow], subsetLabels[icol])
frequency = Table[irow,icol]
cover[irow] = cover[irow] + frequency*isCovered
avgLen[irow] = avgLen[irow] + frequency*intervalSize
}
}
miss = rowFreq-cover; avgLen = avgLen/rowFreq
out=list(avgLen=avgLen,miss=miss,missRate=miss/rowFreq,coverRate=cover/rowFreq)
return(out)
"""


# Given the row from a dataframe, get the order of values if the row was sorted in the increasing order
def get_order(row):
    list_row = list(row)
    # row.values.tolist()
    return np.argsort(list_row)

# print(get_order(pd.DataFrame({'a':[1, 5,3]})))


# Constructs Prediction Intervals and saves it into csv
def category_pred_interval(prob_matrix, labels, prob_value, y_true, filename):
    n_cases = prob_matrix.shape[0]
    pred_list: list = [1] * n_cases
    for i in range(0, n_cases):
        p = prob_matrix[i]
        ip = get_order(p)
        p_ordered = list(p[ip])
        labels_ordered = [labels[i] for i in ip[::-1]]
        g = list(np.cumsum([0] + p_ordered))[::-1]
        k = np.min(np.where(np.array(g) <= (1-prob_value))) - 1
        pred_labels = list(labels_ordered[:k+1])
        pred_list[i] = '.'.join([str(p) for p in pred_labels])
    true_list = list(y_true)
    true_list = [str(p) for p in true_list]
    df_pred_interval = pd.DataFrame(list(zip(true_list, pred_list)), columns=['True', 'Predicted'])
    tab = df_pred_interval.groupby(['True', 'Predicted']).size().unstack().fillna(value=0)
    # tab.to_csv(filename+".csv")
    return tab


# probMatrix = {'1': [4.082367e-10, 1.822106e-15, 1.000000e+00],
#               '2': [1.315919e-03, 9.986605e-01, 2.359995e-05],
#               '3': [9.279982e-01, 7.195826e-02, 4.356354e-05],
#               '4': [4.277739e-08, 5.927599e-10, 1.000000e+00],
#               '5': [9.980396e-01, 1.888962e-03, 7.143053e-05],
#               '6': [2.143369e-08, 1.411895e-10, 1.000000e+00]
#               }
# correctProbMatrix = pd.DataFrame(np.transpose(pd.DataFrame(probMatrix).to_numpy()))
# correctProbMatrix = correctProbMatrix.set_index(pd.Series([12, 43, 42, 23, 54, 28]))
# print(f'Correct Prob Matrix:\n{correctProbMatrix}')
# print(np.array([1.00000001895, 0.99868409995, 0.9986605, 0.0]))
# print(np.where(np.array([1.00000001895, 0.99868409995, 0.9986605, 0.0]) <= 0.5))
# print(np.mean(np.where(np.array([1.00000001895, 0.99868409995, 0.9986605, 0.0]) <= 0.5)))
# print(np.min(np.where(np.array([2, 4, 56, 12]) > 10)))
# print(category_pred_interval(correctProbMatrix, ["B", "D", "P"]))
#

# prediction_list = category_pred_interval(correctProbMatrix, ["B", "D", "P"], 0.5)
# print(prediction_list)
# correct_values = ["B", "D", "P", "B", "D", "P"]
# print(correct_values)
# df = pd.DataFrame(list(zip(correct_values, prediction_list)), columns=("Correct", "Pred"))
# tab = df.groupby(['Correct', 'Pred']).size().unstack()
# tab = tab.fillna(value=0)


def coverage(table):
    nclass = table.shape[0]
    nsubset = table.shape[1]
    labels = table.index.values.tolist()
    subset_labels = table.columns.values.tolist()
    row_freq = np.sum(table, axis=1)
    cov = np.zeros(nclass)
    avg_len = np.zeros(nclass)
    for i in range(nclass):
        for j in range(nsubset):
            n_char = len(re.split("\.", subset_labels[j]))
            interval_size = n_char
            is_covered = [bool(re.search(labels[i], subset_labels[j]))]
            freq = table.iloc[i, j]
            cov[i] = cov[i] + freq * int(np.any(is_covered))
            avg_len[i] = avg_len[i] + freq * interval_size
    miss = row_freq-cov
    avg_len = avg_len/row_freq

    return {'avg_len': avg_len, 'miss':miss, 'miss_rate': miss/row_freq, 'cov_rate':cov/row_freq}


x = pd.DataFrame({'B':[264, 25, 2], 'B.D':[0,1,1], 'D':[11,269,6], 'P':[1,4,288], 'P.D':[0,0,2]})
x.index = ['B', 'D', 'P']
#
y = coverage(x)
print(coverage(x))
