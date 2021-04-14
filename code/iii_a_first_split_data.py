from sklearn.model_selection import train_test_split
import pandas as pd

train = pd.read_pickle('train.pkl')

# random state sets seed
train, holdout = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)

X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']

holdout_X = holdout.drop(['hotel_cluster'], axis=1)
holdout_y = holdout['hotel_cluster']