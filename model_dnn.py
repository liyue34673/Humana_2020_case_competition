# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout 
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve

train_data = pd.read_csv('../data/cleaned_train.csv')


ind_cd_cols = []
for col in train_data.columns:
    if col.endswith('ind') or col.endswith('cd'):
        ind_cd_cols.append(col)
ind_cd_cols = list(set(ind_cd_cols))


sparse_features = ind_cd_cols + ['cons_cmys', 'cons_hhcomp', 'cons_homstat',
                                           'cons_n2029_y', 'cons_n65p_y', 'cons_online_buyer', 'cons_ret_y', 'cons_retail_buyer', 'cons_veteran_y',
                                           'hedis_dia_eye', 'hedis_dia_hba1c_ge9', 'hedis_dia_hba1c_test',
                                           'hedis_dia_ldc_c_control', 'hedis_dia_ldc_c_screen', 'hedis_dia_ma_nephr', 'hlth_pgm_slvrsnkr_par_status',
                                           'mabh_seg', 'rucc_category', 'cons_hcaccprf']
sparse_features = list(set(sparse_features))


dense_features = set(train_data.columns) - set(sparse_features) - set(['transportation_issues', 'person_id_syn'])
dense_features = list(dense_features)

feature_names = sparse_features + dense_features


target = ['transportation_issues']


train, test = train_test_split(train_data, test_size=0.2, random_state=2020)
train_model_input = train[feature_names]
test_model_input = test[feature_names]


model = Sequential([
    Dense(256, input_dim = len(feature_names)),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])


opt = Adam(lr=0.01)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['binary_crossentropy'])


model.fit(x=train_model_input, y=train[target], validation_split=0.1, batch_size=64, epochs=10, class_weight={0:1, 1:3})


pred_ans = model.predict(test_model_input, batch_size=64)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

ans = pd.Series(pred_ans.reshape((-1,)))
trd = 0.5
ans[ans>=trd] = 1
ans[ans<trd] = 0


pd.Series(test[target].transportation_issues).value_counts()



ans.value_counts()

print(classification_report(test[target], ans))


model.summary()





