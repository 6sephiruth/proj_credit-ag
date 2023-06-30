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

# from utils import *

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

# # base, gaussian
# noise_method = 'gaussian'

# Train dataset
df = pd.read_csv('./datasets/creditcard.csv')

x_normal_df = df[df['Class']==0].iloc[:,1:-2]
y_normal_df = df[df['Class']==0].iloc[:,-1]

x_illegal_df = df[df['Class']==1].iloc[:,1:-2]
y_illegal_df = df[df['Class']==1].iloc[:,-1]

train_df = x_normal_df[:-500]
x_test_df = pd.concat([x_normal_df[-500:], x_illegal_df])
y_test_df = pd.concat([y_normal_df[-500:], y_illegal_df])

def get_pred_label(model_pred):
    # IsolationForest 모델 출력 (1:정상, -1:불량(사기)) 이므로 (0:정상, 1:불량(사기))로 Label 변환
    model_pred = np.where(model_pred == 1, 0, model_pred)
    model_pred = np.where(model_pred == -1, 1, model_pred)
    return model_pred


model = IsolationForest(random_state=0).fit(train_df)
pred = model.predict(x_test_df)
pred = get_pred_label(pred)
acc_score = accuracy_score(pred, y_test_df)

exit()
part_datasets = x_train[['V12', 'V28', 'V16', 'V17', 'V3', 'V21', 'V13', 'V11', 'V27', 'V29', 'V18', 'V1', 'V4', 'V8', 'V14','V6','V15']]
#### noise augmentation ####
part_aug = noise_augmentation(noise_method, part_datasets)
x_aug = x_train.add(part_aug, fill_value=0)

x_total = np.concatenate([x_train, x_aug])

# Model 2
model2 = IsolationForest(random_state=0).fit(x_total)
pred = model2.predict(x_validation)
pred = get_pred_label(pred)
acc_score = accuracy_score(pred, y_validation)

print(f"Augmentation Validation Acc : {acc_score}")


test_pred = model2.predict(x_test)
test_pred = get_pred_label(test_pred)
submit = pd.read_csv('./datasets/sample_submission.csv')
submit['Class'] = test_pred
submit.to_csv('./dacon_result/base_aug-shap-10columns.csv', index=False)

exit()


# try:
#     shap_values = pickle.load(open('./shap_values','rb'))
# except Exception:
#     explainer = shap.TreeExplainer(model, x_train, seed=SEED)
#     shap_values = explainer.shap_values(x_train)

#     shap.plots.bar(shap.Explanation(shap_values))

# #     pickle.dump(shap_values, open('./shap_values','wb'))

# explainer = shap.TreeExplainer(model, x_train, seed=SEED)
# shap_values = explainer.shap_values(x_train)
# shap.plots.bar(shap.Explanation(shap_values))

# # plt.show(shap.plots.bar(shap_values))
# plt.savefig('./shap_img.png')