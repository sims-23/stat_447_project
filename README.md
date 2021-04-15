We used roman numerals to organize the code files

1) To download data (not full dataset but is a subset we created), run i_download_data.py
2) Then run ii_a_clean_data.py to clean data
3) Then run ii_b_wrangle_data.py to do the data transformations
4) To rerun plots ii_c_explanatory_analysis.py, change toggle PLOT_GRAPHS = True. 
However, we would suggest due to the time it takes to run, not to change it. 
Figures can be found in figs folder.
5) Next run iii_a_first_split_data.py to get first split for training and holdout set for preliminary results
6) Running iii_b_methodA.py will give preliminary results for logistic regression
7) Running iii_c_methodB.py will give preliminary results for tree methods
8) Running iii_c_methodB.py will give preliminary results for svm.
Toggle GET_BEST_PARAM = False currently as it takes a long time to run, 
but to see the results of doing grid search, set to True
9) Running iv_a_cv.py runs cross validation. Do not run unless you want to wait an extremely long time.
10) Running v_out_of_sample_preds.py. Do not run again unless you want to wait an eternity. 

