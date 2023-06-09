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

# Train dataset
x_train = pd.read_csv('./datasets/train.csv')
x_train = x_train.iloc[:,1:]

# Validation dataset
val_df = pd.read_csv('./datasets/val.csv')
x_validation = val_df.iloc[:,1:31]
y_validation = val_df.iloc[:,31]

# # Test dataset
# test_df = pd.read_csv('test.csv')
# test_df = test_df.iloc[:,1:]

def get_pred_label(model_pred):
    # IsolationForest 모델 출력 (1:정상, -1:불량(사기)) 이므로 (0:정상, 1:불량(사기))로 Label 변환
    model_pred = np.where(model_pred == 1, 0, model_pred)
    model_pred = np.where(model_pred == -1, 1, model_pred)
    return model_pred

model = IsolationForest(random_state=0).fit(x_train)
pred = model.predict(x_validation)
pred = get_pred_label(pred)

acc_score = accuracy_score(pred, y_validation)
print(acc_score)


try:
    shap_values = pickle.load(open('./shap_values','rb'))
except:
    explainer = shap.TreeExplainer(model, x_train, seed=SEED)
    shap_values = explainer.shap_values(x_train)

    pickle.dump(shap_values, open('./shap_values','wb'))

shap.plots.bar(shap_values)
# plt.savefig(shap_img, 'shap_img.png')