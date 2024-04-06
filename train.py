import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

random_forest = RandomForestClassifier(random_state=42, max_depth=55, max_features=2,
                                    min_samples_leaf=5, min_samples_split=8, n_estimators=250)

# Fit the classifier to the data
model = random_forest.fit(X, y)


with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
