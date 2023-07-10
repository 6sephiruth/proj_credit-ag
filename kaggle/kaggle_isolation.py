import os
import shap
import pickle
import random

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

from utils import *

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

SEED = 0

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# base, gaussian
noise_method = 'base'

# Train dataset
df = pd.read_csv('./datasets/creditcard.csv')

x_normal_df = df[df['Class']==0].iloc[:,1:-2]
y_normal_df = df[df['Class']==0].iloc[:,-1]

x_illegal_df = df[df['Class']==1].iloc[:,1:-2]
y_illegal_df = df[df['Class']==1].iloc[:,-1]

train_df = x_normal_df[:-500]

train_columns = train_df.columns

rnd_idx = np.random.choice(len(train_df), 5000, False)
train_df = pd.DataFrame(np.array(train_df)[rnd_idx,:], columns= train_columns)

x_test_df = pd.concat([x_normal_df[-500:], x_illegal_df])
y_test_df = pd.concat([y_normal_df[-500:], y_illegal_df])

### 데이터 preprocessing (*MinMax, Standardization)
# train_df = data_preprocessing('standardization', train_df)
# x_test_df = data_preprocessing('standardization', x_test_df)

model = IsolationForest(random_state=0).fit(train_df)
pred = model.predict(x_test_df)
pred = get_pred_label(pred)
acc_score = accuracy_score(pred, y_test_df)

print(acc_score)
from sklearn.metrics import precision_score
precision = precision_score(y_test_df, pred, pos_label=1)
print(precision)

from sklearn.metrics import recall_score
recall = recall_score(y_test_df, pred, pos_label=1)
print(recall)
print("------------------------------")


# print(train_df.columns)

# try:
#     shap_values = pickle.load(open('./shap_values','rb'))
# except Exception:
#     explainer = shap.TreeExplainer(model, train_df, seed=SEED)
#     shap_values = explainer.shap_values(train_df)
#     pickle.dump(shap_values, open('./shap_values','wb'))

# shap.plots.bar(shap.Explanation(shap_values),max_display=20)

# # plt.show(shap.plots.bar(shap_values))
# plt.savefig('./kaggle_shap_img_20.png')


# iso_mean = np.mean(model.decision_function(train_df))
# iso_std = np.std(model.decision_function(train_df))

part_datasets = train_df[['V15', 'V8', 'V13', 'V11','V2', 'V16', 'V12', 'V27', 'V18', 'V10','V17','V5','V25','V19','V24']]
#### noise augmentation ####
part_aug = noise_augmentation(noise_method, part_datasets)
part_aug = data_preprocessing('standardization', part_aug)

x_aug = train_df.add(part_aug, fill_value=0)
x_aug = x_aug.fillna(0)

x_total = pd.concat([train_df, x_aug])

# Model 2
model2 = IsolationForest(random_state=0).fit(x_total)
pred = model2.predict(x_test_df)
pred = get_pred_label(pred)
acc_score = accuracy_score(pred, y_test_df)
precision = precision_score(y_test_df, pred, pos_label=1)
recall = recall_score(y_test_df, pred, pos_label=1)

print(f"Augmentation Validation Acc : {acc_score}")
print(f"Augmentation Validation Precision : {precision}")
print(f"Augmentation Validation Recall : {recall}")
