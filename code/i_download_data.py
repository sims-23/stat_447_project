import pandas as pd

data = pd.read_csv('data/train.csv', nrows=200000)

def get_df_top_five_clusters(df):
    top_five_most_clusters = df['hotel_cluster'].value_counts()[:5].index.tolist()
    return df[df['hotel_cluster'].isin(top_five_most_clusters)]

data = get_df_top_five_clusters(data)
print(data.shape)

