import pandas as pd

train = pd.read_csv('data/train.csv', nrows=200000)
test = pd.read_csv('data/train.csv', skiprows=range(1, 200000),  nrows=50000)


def get_df_top_five_clusters(df):
    top_five_most_clusters = df['hotel_cluster'].value_counts()[:5].index.tolist()
    return df[df['hotel_cluster'].isin(top_five_most_clusters)]


train = get_df_top_five_clusters(train)
test = get_df_top_five_clusters(test)

train.to_pickle('train_top_5.pkl')
test.to_pickle('test_top_5.pkl')
