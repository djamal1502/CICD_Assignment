import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

#model = LogisticRegression().fit(X, y)

rforest = RandomForestClassifier()
gboost = GradientBoostingClassifier()

# Adding a VotingClassifier classifier with multiple classifiers to boost the score.
model = VotingClassifier(
    estimators=[
        ('rf', rforest), 
        ('gb', gboost)
    ],
).fit(X, y)


with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
