import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from utils import *

SEED = 1
SET_dataset = '6'
SET_model = 'xgb'

if SET_dataset == '1':
    df = pd.read_csv(f'./datasets/1_Dhanush/card_transdata.csv')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

    x_train, y_train = train_df.iloc[:,1:-1], train_df.iloc[:,-1]
    x_test, y_test = test_df.iloc[:,1:-1], test_df.iloc[:,-1]


elif SET_dataset == '2':

    df = pd.read_csv(f'./datasets/2_Joakim/data/creditcard_csv.csv')
    df['Class'] = df['Class'].str.strip("'")
    df['Class'] = df['Class'].astype(int)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

    x_train, y_train = train_df.iloc[:,1:-1], train_df.iloc[:,-1]
    x_test, y_test = test_df.iloc[:,1:-1], test_df.iloc[:,-1]
elif SET_dataset == '3':
    train_df = pd.read_csv(f'./datasets/3_Kartik/fraudTrain.csv')
    test_df = pd.read_csv(f'./datasets/3_Kartik/fraudTest.csv')

    features = ['cc_num' ,'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']

    x_train = train_df[features]
    y_train = train_df['is_fraud']

    x_test = test_df[features]
    y_test = test_df['is_fraud']

elif SET_dataset == '4':
    df = pd.read_csv(f'./datasets/4_Mishra5001/application_data.csv')
elif SET_dataset == '5':
    df = pd.read_csv(f'./datasets/5_Abhishek/cc_info.csv')
    df2 = pd.read_csv(f'./datasets/5_Abhishek/transactions.csv')
elif SET_dataset == '6':
    df = pd.read_csv(f'./datasets/5_Abhishek/cc_info.csv')

print(df2)


exit()



# print(train_df.columns)
# print(train_df.info())
# print(train_df.describe())

# print(train_df['street'].value_counts())

# raw_data['col0']=pd.Categorical(raw_data['col0']).codes



if SET_model == 'xgb':

    xgb_models = xgb.XGBClassifier(n_estimators=200,
                            max_depth=10,
                            learning_rate=0.5,
                            min_child_weight=0,
                            tree_method='gpu_hist',
                            sampling_method='gradient_based',
                            reg_alpha=0.2,
                            reg_lambda=1.5,
                            random_state=SEED)
    xgb_models.fit(x_train, y_train)

    # y_pred = xgb_models.predict(x_train)
    # accuracy = accuracy_score(y_train, y_pred)
    # print("Train accuracy: %.2f" % (accuracy * 100.0))

    # y_pred = xgb_models.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Test accuracy: %.2f" % (accuracy * 100.0))

    evaluate_classification(xgb_models, "XGBoost", x_train, x_test, y_train, y_test)

elif SET_model == 'mlp':

    clf = MLPClassifier(random_state=SEED, max_iter=300).fit(x_train, y_train)

    # y_pred = clf.predict(x_train)
    # accuracy = accuracy_score(y_train, y_pred)
    # print("Train accuracy: %.2f" % (accuracy * 100.0))

    # y_pred = clf.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Test accuracy: %.2f" % (accuracy * 100.0))
    
    evaluate_classification(clf, "mlp", x_train, x_test, y_train, y_test)
