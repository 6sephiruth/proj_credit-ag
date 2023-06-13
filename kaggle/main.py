import os

import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 0

# base, gaussian
noise_method = 'gaussian'


# Train dataset
df = pd.read_csv('./datasets/creditcard.csv')
x_cln_df = df.iloc[:,1:-2]
y_cln_df = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x_cln_df, y_cln_df, test_size=0.2, random_state=SEED)


clf = MLPClassifier(random_state=SEED, max_iter=300).fit(x_train, y_train)

y_pred = clf.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))

y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))





# raw_models = xgb.XGBClassifier(n_estimators=200,
#                         max_depth=10,
#                         learning_rate=0.5,
#                         min_child_weight=0,
#                         tree_method='gpu_hist',
#                         sampling_method='gradient_based',
#                         reg_alpha=0.2,
#                         reg_lambda=1.5,
#                         random_state=SEED)

# raw_models.fit(x_train, y_train)

# y_pred = raw_models.predict(x_train)
# accuracy = accuracy_score(y_train, y_pred)
# print("Train accuracy: %.2f" % (accuracy * 100.0))

# y_pred = raw_models.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Test accuracy: %.2f" % (accuracy * 100.0))