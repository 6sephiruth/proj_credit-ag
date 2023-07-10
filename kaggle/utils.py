import pandas as pd
import numpy as np
from sklearn import preprocessing

def noise_augmentation(method, dataset):

    if method == 'base':
        # noise size = 4
        base_noise = np.random.randint(4, size= dataset.shape)
        aug_dataset = dataset + base_noise

    elif method == 'gaussian':
        mu, sigma = np.min(dataset), np.max(dataset) # mean and standard deviation
        gaussian_noise = np.random.normal(mu, sigma, dataset.shape)
        aug_dataset = dataset + gaussian_noise        

    return aug_dataset

def get_pred_label(model_pred):
    # IsolationForest 모델 출력 (1:정상, -1:불량(사기)) 이므로 (0:정상, 1:불량(사기))로 Label 변환
    model_pred = np.where(model_pred == 1, 0, model_pred)
    model_pred = np.where(model_pred == -1, 1, model_pred)
    return model_pred


def data_preprocessing(method, dataset):

    columns_name = dataset.columns

    if method == 'minmax':

        scaler = preprocessing.MinMaxScaler()
        dataset = scaler.fit_transform(dataset)

        # df로 정리해서 확인
        dataset = pd.DataFrame(dataset, columns=columns_name)

    if method == 'standardization':

        scaler = preprocessing.StandardScaler()
        dataset = scaler.fit_transform(dataset)
            
        dataset = pd.DataFrame(dataset, columns=columns_name)

    return dataset