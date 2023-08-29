import os

import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

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

ebm = ExplainableBoostingClassifier(random_state=SEED)
ebm.fit(x_train, y_train)


# # Global Explanations: What the model learned overall
# ebm_global = ebm.explain_global(name='EBM')
# show(ebm_global)
