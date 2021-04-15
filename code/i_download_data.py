import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

'''
@inputs: pandas dataframe
@outputs: pandas dataframe
@purpose: given this dataset, refactor the dataframe to contain only the rows with the top 5 most frequent 
hotel clusters 
'''
def get_df_top_five_clusters(df):
    top_five_most_clusters = df['hotel_cluster'].value_counts()[:5].index.tolist()
    return df[df['hotel_cluster'].isin(top_five_most_clusters)]


train = get_df_top_five_clusters(train)
test = get_df_top_five_clusters(test)
